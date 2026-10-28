<div align="center">
    <h2>
        BuildFlow: Rethinking Building Extraction from High-Resolution Remote Sensing Images as Deterministic Continuous Evolution
    </h2>
    <p>
        <a href="https://qwerasdzxhjk.github.io">Taochang Liu</a>, <a href="http://www.sysuzxc.site">Xinchang Zhang</a>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=gnUd7ywAAAAJ">Ji Qi<sup>*</sup></a>, <a href="https://geo.gzhu.edu.cn/info/1176/4090.htm">Yongjian Ruan</a>, Linrui Xu, <a href="https://sysu-3dstailab.github.io/3DSTAILab">Yiping Chen</a> and <a href="https://faculty.csu.edu.cn/lihaifeng/zh_CN/index.htm">Haifeng Li</a>
        <br><sup>*</sup> Corresponding author.
    </p>
</div>

<div align="center">
  <img src="resource/main.png" width="800"/>
</div>

# BuildFlow

> **Notice:** This paper has been accepted for publication in *IEEE Transactions on Geoscience and Remote Sensing* on 26 October 2026.

**Code for "BuildFlow: Rethinking Building Extraction from High-Resolution Remote Sensing Images as Deterministic Continuous Evolution"**

This repository contains the official PyTorch implementation of BuildFlow: Rethinking Building Extraction from High-Resolution Remote Sensing Images as Deterministic Continuous Evolution.

## Setup

```bash
git clone https://github.com/qwerasdzxhjk/BuildFlow
cd BuildFlow
conda create -n buildflow python=3.10 -y
conda activate buildflow
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

By default the scripts use CUDA device `0`. Set `BUILD_FLOW_CUDA_VISIBLE_DEVICES` to select another GPU.

## Datasets

* **WHU Building Dataset**: [Download here](https://gpcv.whu.edu.cn/data/building_dataset.html)
* **WHU-Mix Dataset**: [Download here](https://gpcv.whu.edu.cn/data/whu-mix(raster)/whu_mix%20(raster).html)
* **VHIR Dataset**: A UAV dataset described in the paper; please contact the authors to obtain it.

All datasets are tiled into `512x512` images and binary masks with matching filenames. Fill in the dataset paths in `configs/WHU.py`, `configs/WHU_Mix.py`, and `configs/VHIR.py`.

## Pre-trained Models

Before training or testing, download the required foundation models:

* **VAE**: Download `assets.tar` from [here](https://huggingface.co/QHL067/CrossFlow/blob/main/assets.tar) and extract it to the appropriate directory.
* **CrossFlow**: Download `t2i_512px_t5_dit.pth` from [here](https://huggingface.co/QHL067/CrossFlow/blob/main/pretrained_models/t2i_512px_t5_dit.pth).

**BuildFlow weights are not publicly available.** Please contact the authors for access inquiries.

## Data Preparation

Pre-encode the image/mask pairs into VAE latents to accelerate training. Edit the paths in `extract_features.py` (`vae_path`, `source_root`, `target_root`, `save_dir`) and run:

```bash
python extract_features.py
```

Then set `train_latent_root` and `val_latent_root` in the corresponding config file.

## Training

Fill in the dataset paths in the config file, then run:

```bash
python train.py --config=configs/WHU.py
```

Use `configs/WHU_Mix.py` or `configs/VHIR.py` for the other datasets.

## Testing

```bash
python test.py --config=configs/WHU.py
```

The script reports IoU, F1, B-F1, B-SSIM, and B-IoU, and saves the predicted masks to `pred_mask_dir`.

## Polygon Geometry Evaluation

Evaluate the saved masks with polygon-level geometry metrics:

```bash
python evaluate_geometry.py --pred_dir=/path/to/pred_masks --gt_dir=/path/to/test/masks \
    --output_csv=geometry_metrics.csv
```

This reports C-IoU, Vertex P/R/F1@3px, PoLiS, Boundary F@2px, and image IoU. Predictions and ground truth are polygonized with 8-connected components, a minimum area of 16 px^2, and Douglas-Peucker simplification with epsilon = 2.0 px.

## VAE Evaluation

```bash
python test_vae.py --vae_path=/path/to/autoencoder_kl.pth --mask_dir=/path/to/masks
```

## Trajectory Analysis

```bash
python analyze.py --config=configs/WHU.py \
  --input_dir=/path/to/test/images --label_dir=/path/to/test/masks \
  --output_dir=/path/to/output --target_image=image_001
```

`--target_image` is optional and should contain the filename without extension. If omitted, all images in `input_dir` are processed.

## Acknowledgement

This code is built upon the excellent work of the following repositories:

* [CrossFlow](https://github.com/qihao067/CrossFlow)
* [DiT](https://github.com/facebookresearch/DiT)