# DCWNet: Bridging Dual-Codebook Reconstruction and Wavelet Refinement for Low-Light Image Enhancement

[![Paper](https://img.shields.io/badge/Paper-DCWNet-blue.svg)](#)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch 1.13.1](https://img.shields.io/badge/PyTorch-1.13.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> **Kangbiao Shi**<sup>1</sup>, **Zhaokun He**<sup>1</sup>, **Anh-Dzung Doan**<sup>2</sup>, **Li Yan**<sup>1*</sup>, **Qingsen Yan**<sup>1,3*</sup>  
> <sup>1</sup>Northwestern Polytechnical University &nbsp; <sup>2</sup>The University of Adelaide &nbsp; <sup>3</sup>Shenzhen Research Institute of Northwestern Polytechnical University  
> <sup>*</sup>Corresponding Authors

<p align="center">
  <a href="#">📜 Paper</a> |
  <a href="#">🌐 Project Page</a> |
  <a href="#">📦 Pretrained Models</a> |
  <a href="#">📹 Demo</a>
</p>

---

## 📋 Abstract

Low-Light Image Enhancement (LLIE) aims to improve the visual quality of images captured under poor illumination. Existing methods often apply uniform enhancement to the whole image, which may cause color distortion, noise amplification, and uneven exposure. To address these issues, we propose **DCWNet**, a two-branch low-light image enhancement framework built in the **HVI color space**.

DCWNet reconstructs the HV color map with a **dual-codebook strategy**, where a **Bright Codebook** provides stable color prototypes from normal-light images and a **Dark Codebook** adaptively captures chromatic distributions in severely underexposed regions. A **Context-Aware Module (CAM)** and uncertainty-guided fusion further improve spatial consistency and suppress artifacts. For illumination enhancement, DCWNet introduces a **Frequency-Decoupled Wavelet Transform (FDWT)** module, which separately optimizes low- and high-frequency components to preserve details and alleviate local overexposure/underexposure. Extensive experiments on ten benchmark datasets demonstrate that DCWNet achieves state-of-the-art performance.

---

## 🔥 News

- **[2026/05]** Initial repository and README released.
- **[Coming Soon]** Code, pretrained models, and demo will be released.

---

## 📖 Method Overview

DCWNet contains two complementary branches:

- **HV-branch for color restoration**
  - Bright Codebook for reliable normal-light color prototypes
  - Dark Codebook for severely underexposed regions
  - Context-Aware Module for spatially consistent color reconstruction
  - Uncertainty-guided fusion for high-entropy regions

- **I-branch for illumination enhancement**
  - Frequency-Decoupled Wavelet Transform (FDWT)
  - Cross-Frequency Interaction Guided Enhancement (CFIGE)
  - Adaptive Illumination-Aware Low-Frequency Optimizer (AILO)

<p align="center">
  <img src="figs/overview.png" width="900" alt="DCWNet Architecture">
</p>

---

## 🛠️ Requirements

The environment is provided in `dcwnet.yaml`.

Main dependencies:

- Python 3.9
- PyTorch 1.13.1
- Torchvision 0.14.1
- CUDA 11.7 runtime packages
- OpenCV
- NumPy
- SciPy
- scikit-image
- PyWavelets
- LPIPS
- Gradio
- THOP

---

## 🌍 Environment Setup

Create the Conda environment from the provided YAML file:

```bash
conda env create -f dcwnet.yaml
conda activate dcwnet
```

Alternatively, create the environment manually:

```bash
conda create -n dcwnet python=3.9 -y
conda activate dcwnet
pip install torch==1.13.1 torchvision==0.14.1
pip install -r requirements.txt
```

> Note: Please make sure your CUDA driver is compatible with the PyTorch/CUDA version used in this repository.

---

## ⬇️ Dataset Preparation

We evaluate DCWNet on paired and unpaired low-light image enhancement benchmarks.

### Paired Training / Testing Datasets

| Dataset | Description |
|---------|-------------|
| LOLv1 | Paired low-light/normal-light benchmark |
| LOLv2-real | Real paired low-light dataset |
| LOLv2-synthetic | Synthetic paired low-light dataset |
| SICE | Multi-exposure image enhancement dataset |
| SID / Sony-Total-Dark | Extremely dark image enhancement benchmark |

### Unpaired Testing Datasets

| Dataset | Images |
|---------|--------|
| DICM | 69 |
| LIME | 10 |
| MEF | 17 |
| NPE | 8 |
| VV | 24 |

### Recommended Directory Structure

```text
datasets/
├── LOLv1/
│   ├── train/
│   │   ├── low/
│   │   └── high/
│   └── test/
│       ├── low/
│       └── high/
├── LOLv2-real/
│   ├── train/
│   │   ├── low/
│   │   └── high/
│   └── test/
│       ├── low/
│       └── high/
├── LOLv2-synthetic/
│   ├── train/
│   │   ├── low/
│   │   └── high/
│   └── test/
│       ├── low/
│       └── high/
├── SICE/
├── SID/
└── unpaired/
    ├── DICM/
    ├── LIME/
    ├── MEF/
    ├── NPE/
    └── VV/
```

---

## 🚀 Getting Started

### 1. Training

Train DCWNet on a paired dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset LOLv1 \
  --data_root ./datasets/LOLv1 \
  --save_dir ./experiments/dcwnet_lolv1 \
  --batch_size 4 \
  --patch_size 256 \
  --lr 1e-4
```

Train on LOLv2-synthetic:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset LOLv2-synthetic \
  --data_root ./datasets/LOLv2-synthetic \
  --save_dir ./experiments/dcwnet_lolv2_syn \
  --batch_size 4 \
  --patch_size 256 \
  --lr 1e-4
```

Resume training from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset LOLv1 \
  --data_root ./datasets/LOLv1 \
  --save_dir ./experiments/dcwnet_lolv1 \
  --resume ./experiments/dcwnet_lolv1/checkpoints/latest.pth
```

### 2. Testing

Evaluate a trained model on paired test data:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --data_root ./datasets/LOLv1/test \
  --checkpoint ./checkpoints/dcwnet_lolv1.pth \
  --result_dir ./results/LOLv1
```

Evaluate on unpaired datasets:

```bash
CUDA_VISIBLE_DEVICES=0 python test_unpaired.py \
  --input_dir ./datasets/unpaired/DICM \
  --checkpoint ./checkpoints/dcwnet_lolv2_syn.pth \
  --result_dir ./results/DICM
```

### 3. Inference on Custom Images

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --input ./examples/input \
  --checkpoint ./checkpoints/dcwnet.pth \
  --output ./examples/output
```

### 4. Optional Gradio Demo

```bash
python app.py --checkpoint ./checkpoints/dcwnet.pth
```

---

## 📦 Pretrained Models

| Model | Training Dataset | Download |
|-------|------------------|----------|
| DCWNet-LOLv1 | LOLv1 | Coming soon |
| DCWNet-LOLv2-real | LOLv2-real | Coming soon |
| DCWNet-LOLv2-synthetic | LOLv2-synthetic | Coming soon |

Place downloaded checkpoints under:

```text
checkpoints/
├── dcwnet_lolv1.pth
├── dcwnet_lolv2_real.pth
└── dcwnet_lolv2_syn.pth
```

---

## ✨ Qualitative Results

<details>
<summary><strong>LOLv1 Results</strong></summary>
<br>
<p align="center">
  <img src="figs/lolv1.png" width="900" alt="Qualitative results on LOLv1">
</p>
</details>

<details>
<summary><strong>LOLv2 Results</strong></summary>
<br>
<p align="center">
  <img src="figs/lolv2.png" width="900" alt="Qualitative results on LOLv2">
</p>
</details>

<details>
<summary><strong>Unpaired Dataset Results</strong></summary>
<br>
<p align="center">
  <img src="figs/unpaired.png" width="900" alt="Qualitative results on unpaired datasets">
</p>
</details>

---

## 📊 Quantitative Results

### Results on LOLv1 / LOLv2

| Dataset | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|-------|-------|--------|
| LOLv1 | **28.422** | **0.893** | **0.061** |
| LOLv2-real | **24.490** | **0.877** | **0.099** |
| LOLv2-synthetic | **26.840** | **0.942** | **0.038** |

### Results on SICE / Sony-Total-Dark

| Dataset | PSNR↑ | SSIM↑ |
|---------|-------|-------|
| SICE | **13.776** | **0.661** |
| Sony-Total-Dark | **23.962** | **0.708** |

### Results on Unpaired Datasets

| Metric | DICM | LIME | MEF | NPE | VV | Average |
|--------|------|------|-----|-----|----|---------|
| BRISQUE↓ | **16.199** | **16.233** | **11.271** | 13.747 | **20.954** | **16.285** |
| NIQE↓ | 3.704 | **3.778** | **3.402** | 3.864 | **2.643** | **3.481** |

---

## 🧩 Repository Structure

```text
DCWNet/
├── assets/                  # Project assets
├── checkpoints/             # Pretrained checkpoints
├── datasets/                # Dataset root or symlinks
├── figs/                    # Figures used in README
├── models/                  # DCWNet architecture
│   ├── dcwnet.py
│   ├── codebook.py
│   ├── fdwt.py
│   └── modules.py
├── losses/                  # Loss functions
├── utils/                   # Data, metrics, visualization utilities
├── train.py                 # Training script
├── test.py                  # Paired evaluation script
├── test_unpaired.py         # Unpaired evaluation script
├── inference.py             # Inference script
├── app.py                   # Optional Gradio demo
├── dcwnet.yaml              # Conda environment
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📏 Troubleshooting

- **CUDA / PyTorch mismatch**: Check that your GPU driver supports the CUDA runtime installed with PyTorch.
- **Out of memory**: Reduce `--batch_size` or `--patch_size`.
- **Dataset path error**: Ensure `low/` and `high/` folders follow the expected directory structure.
- **Metric mismatch**: Verify that images are saved in the same color range and format as used during evaluation.
- **Missing checkpoint**: Download the pretrained model and place it under `checkpoints/`.

---

## 💖 Acknowledgement

This repository is built upon the low-light image enhancement community and related HVI/color-space, vector-quantization, and wavelet-based restoration works. We sincerely thank the authors of the benchmark datasets and open-source implementations.

---

## 🤝 Citation

If this work is helpful for your research, please cite:

```bibtex
@article{shi2026dcwnet,
  title={DCWNet: Bridging Dual-Codebook Reconstruction and Wavelet Refinement for Low-Light Image Enhancement},
  author={Shi, Kangbiao and He, Zhaokun and Doan, Anh-Dzung and Yan, Li and Yan, Qingsen},
  journal={Manuscript},
  year={2026}
}
```

---

## 🔆 Contact

For questions, please open an issue or contact:

- Kangbiao Shi: kangbiaoshi@mail.nwpu.edu.cn
- Qingsen Yan: qingsenyan@nwpu.edu.cn
