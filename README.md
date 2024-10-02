# AdaFlow: Efficient Long Video Editing via Adaptive Attention Slimming And Keyframe Selection

## Environment

```text
conda env create -f environment.yml
```

## Preprocess

Preprocess you video by running using the following command:

```
python preprocess.py --data_path <data/myvideo.mp4>
```

Additional arguments:

```
                     --save_dir <latents>
                     --H <video height>
                     --W <video width>
                     --sd_version <Stable-Diffusion version>
                     --steps <number of inversion steps>
                     --save_steps <number of sampling steps that will be used later for editing>
                     --n_frames <number of frames>
```

## Editing

To edit your video, first create a yaml config as in `configs/config.yaml`. Then run

```
python run.py
```

