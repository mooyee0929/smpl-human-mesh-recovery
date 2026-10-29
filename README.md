# SMPL Human Mesh Recovery

Tools for recovering 3D human body pose and shape from images and video with the SMPL body model. The repository puts two approaches side by side: **optimization-based fitting**, a PyTorch implementation of SMPLify that fits SMPL pose and shape to 2D keypoints, and **regression-based recovery**, HMR 2.0 from the 4D-Humans project, a ViT-based network that predicts SMPL parameters for every detected person in a single forward pass. It also includes multi-person video tracking with HMR 2.0 and PHALP, a script for rendering SMPL meshes with chosen shape and pose parameters, and helpers for converting OpenPose keypoints and extracting video frames.

## Components

### SMPLify fitting in PyTorch (`fit_smplify.py`)

Fits the gender-neutral SMPL model to one image at a time from 2D joints with confidences:

1. **Camera initialization.** An initial camera depth is estimated from the ratio of 3D to 2D torso length (shoulder-to-hip distances) under a fixed focal length of 5000 px. Camera translation and global body orientation are then refined with Adam for roughly 1,000 steps on the torso-joint reprojection error, with a penalty that keeps the depth near its initial estimate.
2. **Pose and shape optimization.** Starting from the mean pose of an 8-component Gaussian mixture pose prior, the 10 shape coefficients, the 23 body-joint rotations and the global orientation are optimized in four Adam stages that gradually relax the priors:

   | Stage | Pose prior weight | Angle prior weight | Shape prior weight | Learning rate | Steps |
   |:---:|:---:|:---:|:---:|:---:|:---:|
   | 1 | 10.0 | 30.0 | 100.0 | 0.01 | 1,000 |
   | 2 | 5.0 | 10.0 | 50.0 | 0.01 | 1,500 |
   | 3 | 1.0 | 2.0 | 10.0 | 0.005 | 5,000 |
   | 4 | 0.5 | 1.0 | 1.0 | 0.005 | 5,000 |

   The objective is a confidence-weighted 2D reprojection loss plus an angle prior that penalizes unnatural elbow and knee bending, an L2 shape prior on the coefficients, and the max-mixture GMM pose prior. The prior is a PyTorch port of the original chumpy implementation (`smplify_lib/torch_max_mixture_prior.py`).
3. **Keypoint formats.** `--dataset lsp` uses the 14-joint LSP layout (the LSP hip joints get zero weight because they are defined differently from SMPL's); `--dataset openpose` uses the first 15 OpenPose BODY_25 joints. Because SMPL has no head-top joint, mesh vertex 411 stands in for it.
4. **Ablation.** `--reproj_only` drops all priors and runs a single 1,000-step stage on the reprojection loss alone, to study what the priors contribute.

### SMPL rendering (`render_smpl_tpose.py`)

Builds an SMPL mesh from shape coefficients and an axis-angle body pose and renders it with pyrender. The script applies an example arm pose and contains commented-out shape-coefficient presets for exploring how the coefficients change the body.

### HMR 2.0 image demo (`demo.py`, `gradio_app.py`)

Detects people with a Detectron2 ViTDet cascade Mask R-CNN (or a faster RegNetY Mask R-CNN), runs HMR 2.0 on every detection, and renders the predicted mesh over each crop, optionally with side and top views, an all-people composite, and OBJ mesh export. The pretrained HMR 2.0 checkpoint and supporting data are downloaded automatically to `~/.cache/4DHumans` on first run. `gradio_app.py` serves the same pipeline as a web demo.

### Video tracking (`track.py`, `extract_frames.py`)

`track.py` plugs HMR 2.0 into the PHALP tracker (as the 4D-Humans project does) to track people across a frame sequence and render their meshes into a video. `extract_frames.py` splits the rendered video back into individual frames.

### Evaluation and training (`eval.py`, `train.py`)

The HMR 2.0 evaluation and training entry points from 4D-Humans are kept. `eval.py` reports reconstruction error and MPJPE on Human3.6M and 3DPW and 2D keypoint error and PCK on LSP-Extended, PoseTrack and COCO, appending results to a CSV. It needs those datasets at the paths in `hmr2/configs/datasets_eval.yaml`. `train.py` is configured with Hydra (`hmr2/configs_hydra/`) and expects the training tarballs fetched by `fetch_training_data.sh`; the `data` config group it references (`mix_all.yaml`) is not included in this repository.

## Repository structure

```
.
├── hmr2-smplify/                  main code directory (see below)
├── smplify_public/                original SMPLify release (Python 2, chumpy/OpenDR), kept as a
│                                  reference for the PyTorch port
└── slurm_nerf_lego.sh             SLURM job script for a NeRF training run (not used by this project)
```

`hmr2-smplify/`:

```
hmr2-smplify/
├── fit_smplify.py             PyTorch SMPLify fitting
├── render_smpl_tpose.py       render SMPL from shape / pose parameters
├── demo.py                    HMR 2.0 on a folder of images
├── gradio_app.py              HMR 2.0 web demo
├── track.py                   HMR 2.0 + PHALP video tracking (Hydra config)
├── extract_frames.py          video -> frames
├── process_pose_jsons.py      OpenPose JSON -> (3, num_joints) .npy per frame
├── eval.py, train.py          HMR 2.0 evaluation and training
├── fetch_training_data.sh     download HMR 2.0 training tarballs
├── run.sh, checkgpu.sh        SLURM jobs (tracking run, GPU check)
├── environment.yml, setup.py
├── hmr2/                      HMR 2.0 package: ViT backbone, transformer SMPL head,
│                              SMPL wrapper, datasets, renderer, configs
├── smplify_lib/               max-mixture pose prior (original and PyTorch versions)
├── assets/                    teaser image from 4D-Humans
└── example_data/
    ├── images/lsp/            5 LSP images with 14-joint 2D keypoints (*_joints.npy)
    └── videos/example_1/      23-frame clip: source .avi, frames, OpenPose JSON, joints .npy
```

## Setup

Two dependency specifications are included. `environment.yml` creates a conda environment named `4D-humans` (Python 3.10, PyTorch with CUDA 11.8, Detectron2, chumpy, smplx, pyrender and the Hydra/Lightning training stack). `setup.py` installs the `hmr2` package with pinned versions (torch 1.13.1, pytorch-lightning 1.9.5, timm 0.6.12); the `all` extra adds Detectron2.

```bash
conda env create -f environment.yml
conda activate 4D-humans
pip install -e ".[all]"
```

Run these from `hmr2-smplify/`. `track.py` additionally imports the PHALP tracker (`phalp`) and `neural_renderer`, and `gradio_app.py` needs `gradio`; none of these are listed in the dependency files.

### SMPL model files

The SMPL and SMPLify model files are licensed separately by the Max Planck Institute for Intelligent Systems for non-commercial research use and are not included in this repository. Register and download them from [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) and [smplify.is.tue.mpg.de](https://smplify.is.tue.mpg.de/), then place them in `hmr2-smplify/data/`:

```
hmr2-smplify/data/
├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl   gender-neutral SMPL model
└── gmm_08.pkl                                   8-component pose prior
```

HMR 2.0 also looks for the neutral model at `data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` and converts it to the format it needs under `~/.cache/4DHumans` automatically.

The reference SMPLify release in `smplify_public/` reads its model files from `smplify_public/code/models/`: the SMPL `basicModel_*_lbs_10_207_0_v1.0.0.pkl` files, plus `gmm_08.pkl` and `regressors_locked_normalized_{female,male,hybrid}.npz` from the SMPLify download. These paths are listed in `.gitignore` so the licensed files are not committed.

## Usage

All commands are run from `hmr2-smplify/`:

```bash
cd hmr2-smplify
```

Fit SMPL to the LSP examples (writes to `results/lsp/smplify/`):

```bash
python fit_smplify.py --viz
python fit_smplify.py --viz --selected_indices 0,2          # only some images
python fit_smplify.py --reproj_only --out_dir results/lsp/smplify_reproj_only --viz
```

Fit SMPL to OpenPose keypoints from the example video:

```bash
# Convert OpenPose JSON to per-frame joint arrays (already provided in joints/)
python process_pose_jsons.py \
    --input_dir example_data/videos/example_1/openpose \
    --output_dir example_data/videos/example_1/joints

python fit_smplify.py --dataset openpose \
    --img_dir example_data/videos/example_1/images \
    --joints_dir example_data/videos/example_1/joints \
    --out_dir results/example_1/smplify --viz
```

For each image, `fit_smplify.py` creates a numbered folder containing a pickle with the fitted camera translation, focal length, pose, shape coefficients, projected joints and loss history, plus a grid of loss curves per stage. With `--viz` it also saves the input with ground-truth and fitted 2D joints next to the mesh overlay, and snapshots of the camera initialization and each optimization stage. Other options: `--model_dir` (default `data/`), `--n_betas`, `--flength`.

Render an SMPL mesh:

```bash
python render_smpl_tpose.py --out results/smpl_tpose.png --res 512
```

Run HMR 2.0 on images:

```bash
python demo.py --img_folder example_data/images/lsp --out_folder demo_out \
    --side_view --top_view --full_frame --save_mesh
python demo.py --img_folder example_data/images/lsp --detector regnety   # faster detector
```

Track people through the example frame sequence and split the rendered video into frames (the same steps as `run.sh`):

```bash
python track.py video.source="./example_data/videos/example_1/images" \
    video.output_dir=results/example_1/hmr2
python extract_frames.py --video results/example_1/hmr2/PHALP_images.mp4 \
    --out_dir results/example_1/hmr2/extracted_frames
```

Evaluate HMR 2.0 (requires the evaluation datasets):

```bash
python eval.py --dataset 3DPW-TEST,LSP-EXTENDED --results_file results/eval_regression.csv
```

No result images are committed; outputs are written to `results/` and `demo_out/`, which are git-ignored.

## Acknowledgements

- HMR 2.0 and the `hmr2` package, `demo.py`, `track.py`, `train.py`, `eval.py`, `gradio_app.py` and `fetch_training_data.sh` come from [4D-Humans](https://github.com/shubham-goel/4D-Humans) (Goel et al., "Humans in 4D: Reconstructing and Tracking Humans with Transformers", ICCV 2023), released under the MIT License (see [`hmr2-smplify/LICENSE.md`](hmr2-smplify/LICENSE.md)). Video tracking uses [PHALP](https://github.com/brjathu/PHALP), and person detection uses [Detectron2](https://github.com/facebookresearch/detectron2).
- The fitting procedure, joint mappings and the chumpy pose prior (`smplify_lib/max_mixture_prior.py`) follow SMPLify (Bogo et al., "Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image", ECCV 2016). `smplify_public/` is the original SMPLify code release, provided by the Max Planck Society for research purposes only under the [SMPLify license](http://smplify.is.tue.mpg.de/license).
- SMPL body model: Loper et al., "SMPL: A Skinned Multi-Person Linear Model", SIGGRAPH Asia 2015. Model files are subject to the SMPL license and must be obtained from the official website.
- The still-image examples in `hmr2-smplify/example_data/images/lsp/` come from the Leeds Sports Pose (LSP) dataset.
