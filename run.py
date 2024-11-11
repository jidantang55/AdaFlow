import glob
import math
import os
import random

import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline
import matplotlib.pyplot as plt

from adaflow_utils import *
from util import save_video, seed_everything
from dift_sd import SDFeaturizer

# suppress partial model loading warning
logging.set_verbosity_error()

VAE_BATCH_SIZE = 10


def draw_similarity_map(sim_tensor, save_path):
    sim_tensor = sim_tensor.cpu().numpy()
    sim_tensor = (sim_tensor * 255).astype(np.uint8)
    sim_tensor = cv2.applyColorMap(sim_tensor, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, sim_tensor)


def draw_kv(h, w, key_matching, save_path, base_images):
    n = len(key_matching)
    # n_cols = int(np.ceil(np.sqrt(n)))
    n_cols = 6
    n_rows = int(np.ceil(n / n_cols))

    for i in range(n):
        images = np.zeros((n, h, w), dtype=np.float32)
        for idx in key_matching[i]:
            img_idx = idx // (w * h)
            pixel_idx = idx % (w * h)
            row = pixel_idx // w
            col = pixel_idx % w
            images[img_idx, row, col] = 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
        axs = axs.flatten()

        for j in range(n):
            base_image = base_images[j].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            base_image = cv2.resize(base_image, (w, h), interpolation=cv2.INTER_LINEAR)
            masked_image = np.zeros_like(base_image)
            for c in range(3):
                masked_image[:, :, c] = base_image[:, :, c] * images[j]

            axs[j].imshow(masked_image)
            axs[j].set_title(f'Image {j}', fontsize=16)
            axs[j].axis('off')

        for j in range(n, len(axs)):
            fig.delaxes(axs[j])

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(f'{save_path}/image_{h}_{w}_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)


class AdaFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]

        sd_version = config["sd_version"]
        self.sd_version = sd_version
        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        # pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
        print('SD model loaded')

        # data
        self.latents_path = self.get_latents_path()
        # load frames
        self.paths, self.frames, self.latents, self.eps = self.get_data()
        if self.sd_version == 'depth':
            self.depth_maps = self.prepare_depth_maps()

        self.text_embeds = self.get_text_embeds(config["prompt"], config["negative_prompt"])
        pnp_inversion_prompt = self.get_pnp_inversion_prompt()
        self.pnp_guidance_embeds = self.get_text_embeds(pnp_inversion_prompt, pnp_inversion_prompt).chunk(2)[0]
        self.matching, self.batch_begin_idx, self.dift_size, self.key_matching, self.all_pivotal_idx = self.prepare_dift_matching(model_key)

    @torch.no_grad()
    def prepare_depth_maps(self, model_type='DPT_Large', device='cuda'):
        depth_maps = []
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        for i in range(len(self.paths)):
            img = cv2.imread(self.paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            latent_h = img.shape[0] // 8
            latent_w = img.shape[1] // 8

            input_batch = transform(img).to(device)
            prediction = midas(input_batch)

            depth_map = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(latent_h, latent_w),
                mode="bicubic",
                align_corners=False,
            )
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
            depth_maps.append(depth_map)

        return torch.cat(depth_maps).to(torch.float16).to(self.device)

    def prepare_dift_matching(self, model_key, device='cuda'):
        dift = SDFeaturizer(model_key)
        H = (math.ceil(self.config["H"] / 64), math.ceil(self.config["H"] / 32), math.ceil(self.config["H"] / 16),
             math.ceil(self.config["H"] / 8))
        W = (math.ceil(self.config["W"] / 64), math.ceil(self.config["W"] / 32), math.ceil(self.config["W"] / 16),
             math.ceil(self.config["W"] / 8))
        ft_list = []
        matching = {}
        key_matching = {}
        for i in tqdm(range(len(self.frames)), desc='Extracting DIFT features'):
            ft = dift.forward(self.frames[i], prompt='', t=0, up_ft_index=[2], ensemble_size=8)  # [1, c, h, w]
            ft_list.append(ft[2].detach().cpu())
        if self.config["batch_size"] == "auto":
            batch_begin_idx = [0]
            frames_ft = torch.cat(ft_list)  # [n_frames, c, h, w]
            frames_ft = frames_ft.permute(0, 2, 3, 1)  # [n_frames, h, w, c]
            dim = frames_ft.shape[-1]
            cur_frame_idx = 0
            h, w = frames_ft[0].shape[:2]
            while cur_frame_idx < len(frames_ft):
                ft1 = frames_ft[cur_frame_idx].reshape(-1, dim).to(device)  # [seq_len, c]
                ft1 = ft1 / ft1.norm(dim=-1, keepdim=True)
                for frame_idx in range(cur_frame_idx + 1, len(frames_ft)):
                    # print(f"Matching {cur_frame_idx} with {frame_idx}")
                    ft2 = frames_ft[frame_idx].reshape(-1, dim).to(device)  # [seq_len, c]
                    ft2 = ft2 / ft2.norm(dim=-1, keepdim=True)
                    similarity = ft1 @ ft2.T  # [seq_len, seq_len]
                    max_sim = similarity.max(dim=1).values  # [seq_len]
                    max_sim = max_sim.reshape(h, w)
                    if not self.check_similarity(max_sim, max(h, w) // 16,
                                                 max(h, w) // 32) or frame_idx - cur_frame_idx >= self.config[
                        "max_batch_size"] - 1:
                        draw_similarity_map(max_sim,
                                            f'{self.config["output_path"]}/similarity_{cur_frame_idx}_{frame_idx}.png')
                        batch_begin_idx.append(frame_idx)
                        cur_frame_idx = frame_idx + 1
                        if cur_frame_idx == len(frames_ft) - 1:
                            cur_frame_idx += 1
                        break
                    if frame_idx == len(frames_ft) - 1:
                        cur_frame_idx = frame_idx + 1
        else:
            batch_begin_idx = list(range(0, len(self.frames), self.config["batch_size"]))
        torch.cuda.empty_cache()
        print(f'Batch begin index: {batch_begin_idx}')
        if len(batch_begin_idx) > self.config["max_kv_size"]:
            keep_ratio = self.config["max_kv_size"] / len(batch_begin_idx)
            print(f"Will only keep {100 * keep_ratio:.2f}% of K/V when editing key frames!")
        all_pivotal_idx = self.generate_pivotal(batch_begin_idx)
        for (h, w) in tqdm(zip(H, W), desc='Matching DIFT features', total=len(H)):
            # print(f"Matching for {h}x{w}")
            frames_ft = torch.cat(ft_list)  # [n_frames, c, h, w]
            frames_ft = nn.functional.interpolate(frames_ft, size=(h, w), mode='bilinear', align_corners=False)
            frames_ft = frames_ft.permute(0, 2, 3, 1)  # [n_frames, h, w, c]
            for i, frame_idx in enumerate(batch_begin_idx):
                batch_size = batch_begin_idx[i + 1] - frame_idx if i + 1 < len(batch_begin_idx) else len(
                    frames_ft) - frame_idx
                matching[f'{h}_{w}_{i}'] = self.get_matching(frames_ft, frame_idx, batch_size, device=device).detach().cpu()
            torch.cuda.empty_cache()
            if len(batch_begin_idx) > self.config["max_kv_size"]:
                for t in self.scheduler.timesteps:
                    t = int(t)
                    key_frames_ft = frames_ft[all_pivotal_idx[t]]
                    key_match = self.get_matching(key_frames_ft, 0, len(key_frames_ft), is_key=True, device=device)
                    k = int(key_match.shape[1] // (key_match.shape[0] / self.config["max_kv_size"]))
                    key_matching[f'{h}_{w}_{t}'] = torch.topk(key_match, k, dim=1, largest=True).indices.detach().cpu()
                    # if t == 981 and h == 64 and w == 64:
                    #     draw_kv(h, w, key_matching[f'{h}_{w}_{t}'], self.config["output_path"], self.frames[all_pivotal_idx[t]])
        torch.cuda.empty_cache()
        return matching, batch_begin_idx, list(zip(H, W)), key_matching, all_pivotal_idx

    def get_matching(self, frames_ft, begin_idx, batch_size, is_key=False, device='cuda'):
        dim = frames_ft.shape[-1]
        if batch_size <= self.config["max_kv_size"]:
            batch_ft = frames_ft[begin_idx:begin_idx + batch_size].reshape(-1, dim).to(device)  # [batch_size * seq_len, c]
            batch_ft = batch_ft / batch_ft.norm(dim=-1, keepdim=True)
            similarity = batch_ft @ batch_ft.T  # [batch_size * seq_len, batch_size * seq_len]
            sim_list = similarity.chunk(batch_size, dim=1)  # [batch_size * seq_len, seq_len]
            idx = []
            for sim in sim_list:
                if not is_key:
                    idx.append(sim.argmax(dim=1).unsqueeze(0))  # [1, batch_size * seq_len]
                else:
                    idx.append(sim.max(dim=1).values.unsqueeze(0))  # [1, batch_size * seq_len]
        else:  # If batch_size is too large, split it into smaller batches. Otherwise, it may cause OOM.
            batch_ft = frames_ft[begin_idx:begin_idx + batch_size].reshape(-1, dim).to(device)  # [batch_size * seq_len, c]
            batch_ft = batch_ft / batch_ft.norm(dim=-1, keepdim=True)
            idx = []
            small_batch_size = self.config["max_batch_size"] // batch_size
            for j in range(0, batch_size, small_batch_size):
                if j + small_batch_size < batch_size:
                    batch_ft_j = frames_ft[begin_idx + j:begin_idx + j + small_batch_size].reshape(-1,
                                                                                                   dim).to(device)  # [batch_size_j * seq_len, c]
                else:
                    batch_ft_j = frames_ft[begin_idx + j:begin_idx + batch_size].reshape(-1, dim).to(device)
                batch_ft_j = batch_ft_j / batch_ft_j.norm(dim=-1, keepdim=True)
                similarity = batch_ft_j @ batch_ft.T  # [batch_size_j * seq_len, batch_size * seq_len]
                sim_list = similarity.chunk(batch_size, dim=1)  # [batch_size_j * seq_len, seq_len]
                if len(idx) == 0:
                    for sim in sim_list:
                        if not is_key:
                            idx.append(sim.argmax(dim=1).unsqueeze(0))
                        else:
                            idx.append(sim.max(dim=1).values.unsqueeze(0))
                else:
                    for k, sim in enumerate(sim_list):
                        if not is_key:
                            idx[k] = torch.cat([idx[k], sim.argmax(dim=1).unsqueeze(0)], dim=1)
                        else:
                            idx[k] = torch.cat([idx[k], sim.max(dim=1).values.unsqueeze(0)], dim=1)
        return torch.cat(idx)  # [batch_size, batch_size * seq_len]

    def check_similarity(self, sim_tensor, region_size, stride):
        if torch.mean(sim_tensor) < self.config["similarity_mean_threshold"]:
            print(f"Mean similarity is too low: {torch.mean(sim_tensor)}")
            return False
        for i in range(0, sim_tensor.shape[0] - region_size + 1, stride):
            for j in range(0, sim_tensor.shape[1] - region_size + 1, stride):
                region_mean = torch.mean(sim_tensor[i:i + region_size, j:j + region_size])
                if region_mean < self.config["similarity_region_threshold"]:
                    print(f"Region mean similarity is too low: {region_mean}")
                    return False
        return True

    def generate_pivotal(self, batch_begin_idx):
        all_pivotal_idx = {}
        for t in self.scheduler.timesteps:
            t = int(t)
            pivotal_idx = []
            for i in range(len(batch_begin_idx)):
                if i + 1 < len(batch_begin_idx):
                    pivotal_idx.append(random.randint(batch_begin_idx[i], batch_begin_idx[i + 1] - 1))
                else:
                    pivotal_idx.append(random.randint(batch_begin_idx[i], len(self.frames) - 1))
            pivotal_idx = torch.tensor(pivotal_idx)
            all_pivotal_idx[t] = pivotal_idx
        return all_pivotal_idx

    def get_pnp_inversion_prompt(self):
        inv_prompts_path = os.path.join(str(Path(self.latents_path).parent), 'inversion_prompt.txt')
        # read inversion prompt
        with open(inv_prompts_path, 'r') as f:
            inv_prompt = f.read()
        return inv_prompt

    def get_latents_path(self):
        latents_path = os.path.join(config["latents_path"], f'sd_{config["sd_version"]}',
                                    Path(config["data_path"]).stem, f'steps_{config["n_inversion_steps"]}')
        latents_path = [x for x in glob.glob(f'{latents_path}/*') if '.' not in Path(x).name]
        n_frames = [int([x for x in latents_path[i].split('/') if 'nframes' in x][0].split('_')[1]) for i in
                    range(len(latents_path))]
        latents_path = latents_path[np.argmax(n_frames)]
        if self.config["n_frames"] == "all":
            self.config["n_frames"] = max(n_frames)
        else:
            self.config["n_frames"] = min(max(n_frames), config["n_frames"])
        if self.config["batch_size"] != "auto":
            if self.config["n_frames"] % self.config["batch_size"] != 0:
                # make n_frames divisible by batch_size
                self.config["n_frames"] = self.config["n_frames"] - (
                        self.config["n_frames"] % self.config["batch_size"])
        print("Number of frames: ", self.config["n_frames"])
        return os.path.join(latents_path, 'latents')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=VAE_BATCH_SIZE, deterministic=False):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents, batch_size=VAE_BATCH_SIZE):
        latents = 1 / 0.18215 * latents
        imgs = []
        for i in range(0, len(latents), batch_size):
            imgs.append(self.vae.decode(latents[i:i + batch_size].to(self.device)).sample.detach().cpu())
        imgs = torch.cat(imgs)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def get_data(self):
        # load frames
        paths = [os.path.join(config["data_path"], "%05d.jpg" % idx) for idx in
                 range(self.config["n_frames"])]
        if not os.path.exists(paths[0]):
            paths = [os.path.join(config["data_path"], "%05d.png" % idx) for idx in
                     range(self.config["n_frames"])]
        frames = [Image.open(paths[idx]).convert('RGB') for idx in range(self.config["n_frames"])]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(self.device)
        save_video(frames, f'{self.config["output_path"]}/input_fps10.mp4', fps=10)
        save_video(frames, f'{self.config["output_path"]}/input_fps20.mp4', fps=20)
        save_video(frames, f'{self.config["output_path"]}/input_fps30.mp4', fps=30)
        # encode to latents
        latents = self.encode_imgs(frames, deterministic=True).to(torch.float16).to(self.device)
        # get noise
        eps = self.get_ddim_eps(latents, range(self.config["n_frames"])).to(torch.float16)
        frames = frames.detach().cpu()
        latents = latents.detach().cpu()
        torch.cuda.empty_cache()
        return paths, frames, latents, eps

    def get_ddim_eps(self, latent, indices):
        noisest = max([int(x.split('_')[-1].split('.')[0]) for x in
                       glob.glob(os.path.join(self.latents_path, f'noisy_latents_*.pt'))])
        latents_path = os.path.join(self.latents_path, f'noisy_latents_{noisest}.pt')
        noisy_latent = torch.load(latents_path)[indices].to(self.device)
        alpha_prod_T = self.scheduler.alphas_cumprod[noisest]
        mu_T, sigma_T = alpha_prod_T ** 0.5, (1 - alpha_prod_T) ** 0.5
        eps = (noisy_latent - mu_T * latent) / sigma_T
        return eps

    @torch.no_grad()
    def denoise_step(self, x, t, indices):
        # register the time step and features in pnp injection modules
        source_latents = load_source_latents_t(t, self.latents_path)[indices]
        latent_model_input = torch.cat([source_latents] + ([x] * 2))
        if self.sd_version == 'depth':
            latent_model_input = torch.cat([latent_model_input, torch.cat([self.depth_maps[indices]] * 3)], dim=1)

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds.repeat(len(indices), 1, 1),
                                      torch.repeat_interleave(self.text_embeds, len(indices), dim=0)])
        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    @torch.autocast(dtype=torch.float16, device_type='cuda')
    def batched_denoise_step(self, x, t, indices, batch_begin_idx):
        denoised_latents = []
        pivotal_idx = self.all_pivotal_idx[int(t)]

        register_pivotal(self, True, pivotal_idx)
        x_pivot = x[pivotal_idx].to(self.device)
        self.denoise_step(x_pivot, t, indices[pivotal_idx])
        torch.cuda.empty_cache()
        register_pivotal(self, False, pivotal_idx)
        for i, b in enumerate(batch_begin_idx):
            batch_size = batch_begin_idx[i + 1] - b if i + 1 < len(batch_begin_idx) else len(x) - b
            register_batch_idx(self, i, batch_begin_idx[i])
            if batch_size <= 2 * self.config["max_kv_size"]:
                register_frame_idx(self, 0, batch_size)
                denoised_latents.append(self.denoise_step(x[b:b + batch_size].to(self.device), t, indices[b:b + batch_size]).detach().cpu())
            else:
                for j in range(0, batch_size, 2 * self.config["max_kv_size"]):
                    if j + int(2 * self.config["max_kv_size"]) < batch_size:
                        register_frame_idx(self, j, j + int(2 * self.config["max_kv_size"]))
                        denoised_latents.append(self.denoise_step(x[b + j:b + j + int(2 * self.config["max_kv_size"])].to(self.device), t, indices[b + j:b + j + int(2 * self.config["max_kv_size"])]).detach().cpu())
                    else:
                        register_frame_idx(self, j, batch_size)
                        denoised_latents.append(self.denoise_step(x[b + j:b + batch_size].to(self.device), t, indices[b + j:b + batch_size]).detach().cpu())
        torch.cuda.empty_cache()

        denoised_latents = torch.cat(denoised_latents)
        return denoised_latents

    def init_method(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_extended_attention_pnp(self, self.qk_injection_timesteps, self.config["max_kv_size"])
        register_conv_injection(self, self.conv_injection_timesteps)
        set_adaflow(self.unet)

    def save_vae_recon(self):
        os.makedirs(f'{self.config["output_path"]}/vae_recon', exist_ok=True)
        decoded = self.decode_latents(self.latents)
        for i in range(len(decoded)):
            T.ToPILImage()(decoded[i]).save(f'{self.config["output_path"]}/vae_recon/%05d.png' % i)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_10.mp4', fps=10)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_20.mp4', fps=20)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_30.mp4', fps=30)
        torch.cuda.empty_cache()

    def edit_video(self):
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        self.save_vae_recon()
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_method(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        noisy_latents = self.scheduler.add_noise(self.latents.to(self.device), self.eps.to(self.device), self.scheduler.timesteps[0])
        self.latents = self.latents.detach().cpu()
        self.eps = self.eps.detach().cpu()
        noisy_latents = noisy_latents.detach().cpu()
        torch.cuda.empty_cache()
        edited_frames = self.sample_loop(noisy_latents, torch.arange(self.config["n_frames"]))
        save_video(edited_frames, f'{self.config["output_path"]}/adaflow_fps_10.mp4')
        save_video(edited_frames, f'{self.config["output_path"]}/adaflow_fps_20.mp4', fps=20)
        save_video(edited_frames, f'{self.config["output_path"]}/adaflow_fps_30.mp4', fps=30)
        print('Done!')

    def sample_loop(self, x, indices):
        register_matching(self, self.matching, self.dift_size, self.key_matching)
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
            x = self.batched_denoise_step(x, t, indices, self.batch_begin_idx)

        decoded_latents = self.decode_latents(x)
        for i in range(len(decoded_latents)):
            T.ToPILImage()(decoded_latents[i]).save(f'{self.config["output_path"]}/img_ode/%05d.png' % i)

        return decoded_latents


def run(config):
    seed_everything(config["seed"])
    print(config)
    editor = AdaFlow(config)
    editor.edit_video()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config-field.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["output_path"] = os.path.join(config["output_path"] + f'_pnp_SD_{config["sd_version"]}',
                                         Path(config["data_path"]).stem,
                                         config["prompt"][:240],
                                         f'attn_{config["pnp_attn_t"]}_f_{config["pnp_f_t"]}',
                                         f'batch_size_{str(config["batch_size"])}',
                                         str(config["n_timesteps"]),
                                         )
    os.makedirs(config["output_path"], exist_ok=True)
    assert os.path.exists(config["data_path"]), "Data path does not exist"
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    run(config)
