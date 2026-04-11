<div align="center">
    <h2>
        BuildFlow: Rethinking Building Extraction from High-Resolution Remote Sensing Imagery as Deterministic Continuous Evolution
    </h2>
    <p>
        <a href="https://qwerasdzxhjk.github.io">Taochang Liu</a>, <a href="http://www.sysuzxc.site">Xinchang Zhang</a>, <a href="https://geo.gzhu.edu.cn/info/1174/2184.htm">Jianzhou Gong<sup>*</sup></a>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=gnUd7ywAAAAJ">Ji Qi<sup>*</sup></a> <a href="https://geo.gzhu.edu.cn/info/1176/4090.htm">Yongjian Ruan</a> Linrui Xu <a href="https://faculty.csu.edu.cn/lihaifeng/zh_CN/index.htm">Haifeng Li</a>
    </p>
</div>

<div align="center">
  <img src="resource/main.png" width="800"/>
</div>

# BuildFlow

**Code for “BuildFlow: Rethinking Building Extraction from High-Resolution Remote Sensing Imagery as Deterministic Continuous Evolution”**

This repository contains the official PyTorch implementation of BuildFlow: Rethinking Building Extraction from High-Resolution Remote Sensing Imagery as Deterministic Continuous Evolution.

## Setup

First, download and set up the repository:

```bash
git clone https://github.com/qwerasdzxhjk/BuildFlow
cd BuildFlow
```

Create and activate a Conda environment:

```bash
conda create -n buildflow python=3.10 -y
conda activate buildflow
```

Install the required dependencies using `uv` for faster installation:

```bash
pip install uv
uv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

## Datasets

Please download the required datasets from their official sources:

* **WHU Building Dataset**: [Download here](https://gpcv.whu.edu.cn/data/building_dataset.html)
* **WHU-Mix Dataset**: [Download here](https://gpcv.whu.edu.cn/data/whu-mix(raster)/whu_mix%20(raster).html)

After downloading, please ensure the datasets are placed in your designated data directory and update the corresponding paths in the `configs/` files.

## Pre-trained Models

Before training or testing, you need to download the necessary foundation models and pre-trained weights.

**Foundation Models:**
* **VAE**: Download `assets.tar` from [here](https://huggingface.co/QHL067/CrossFlow/blob/main/assets.tar) and extract it to the appropriate directory.
* **CrossFlow**: Download the pre-trained weights `t2i_512px_t5_dit.pth` from [here](https://huggingface.co/QHL067/CrossFlow/blob/main/pretrained_models/t2i_512px_t5_dit.pth).

**BuildFlow Weights (Ours):**
* **BuildFlow on WHU**: [Coming soon]()
* **BuildFlow on WHU-Mix**: [Coming soon]()

## Data Preparation

To accelerate the training process, it is highly recommended to pre-process the dataset images using the VAE. This step extracts and caches the latent features:

```bash
python extract_features.py
```

## Training

Once the features are extracted, you can start training the BuildFlow model. We provide configuration files for different datasets (e.g., WHU Building Dataset). 

To train the model using the WHU dataset configuration, run:

```bash
python train.py --config=configs/WHU.py
```

## Testing

We provide testing scripts for both the overall building extraction performance and the specific VAE encoding/decoding capabilities mentioned in the paper.

**1. Evaluate BuildFlow**
To test the trained model for building extraction on the target dataset:

```bash
python test.py --config=configs/WHU.py
```

## Evaluate VAE Performance

To specifically test the VAE encoding and decoding capabilities as described in the paper, run the script without the config argument:

```bash
python test.py
```

## Trajectory & Interpolation Analysis
To intuitively understand the deterministic trajectory evolution and manifold deviations mentioned in the introduction of our paper, we provide a dedicated analysis script. This script compares Pixel Linear and Latent Linear (VAE) interpolations, extracts Shannon entropy and structural clarity metrics, and saves intermediate visual frames along with a CSV report and trend charts.

```bash
python analyze_intro.py --config=configs/WHU.py --target_image=image_001
```

> **⚠️ Important Note:** The `--target_image` parameter should **only contain the filename without the file extension** (e.g., use `image_001`, NOT `image_001.tif` or `image_001.png`).

## Acknowledgement

This code is built upon the excellent work of the following repositories:

* [CrossFlow](https://github.com/qihao067/CrossFlow)
* [DiT](https://github.com/facebookresearch/DiT)