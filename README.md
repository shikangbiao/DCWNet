# DCWNet: Bridging Dual-Codebook Reconstruction and Wavelet Refinement for Low-Light Image Enhancement

[![Paper](https://img.shields.io/badge/Paper-DCWNet-blue.svg)](#)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch 1.13.1](https://img.shields.io/badge/PyTorch-1.13.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> **Kangbiao Shi**<sup>1</sup>, **Zhaokun He**<sup>1</sup>, **Anh-Dzung Doan**<sup>2</sup>, **Li Yan**<sup>1*</sup>, **Qingsen Yan**<sup>1,3*</sup>  
> <sup>1</sup>Northwestern Polytechnical University &nbsp; <sup>2</sup>The University of Adelaide &nbsp; <sup>3</sup>Shenzhen Research Institute of Northwestern Polytechnical University> <sup>*</sup>Corresponding Authors

<p align="center">
  <a href="#">📜 Paper</a> |
  <a href="#">🌐 Project Page</a> |
  <a href="#">📦 Pretrained Models</a> |
  <a href="#">📹 Demo</a>
</p>

---

## 📋 Abstract


Low-Light Image Enhancement (LLIE) aims to improve the visual quality of images degraded by poor illumination. Existing LLIE methods often apply uniform enhancement across the entire image, ignoring the intricate coupling between illumination and color, which leads to color distortion and uneven exposure. Recent HVI-based approaches decouple illumination and color, but they still struggle in extremely dark regions, where the chroma signal is severely attenuated and dominated by noise, resulting in reduced color discriminability and residual artifacts. 
To address these challenges, we introduce **DCWNet**, a novel two-branch LLIE framework in the HVI color space. Specifically, the HV-branch reconstructs the HV color map using a **dual-codebook** strategy: a Bright Codebook for stable, natural-light colors, and a Dark Codebook that adaptively captures chromatic distributions in dark regions. The **Context-Aware Module** further enhances spatial consistency, while an **uncertainty-guided fusion mechanism** focuses on high-entropy regions, jointly suppressing noise and correcting global color bias. In the I-branch, a **Frequency-Decoupled Wavelet Transform** module decomposes the intensity map into high- and low-frequency components for separate processing. By preserving fine details and texture information, it effectively alleviates local overexposure and underexposure through region-aware brightness optimization. Extensive experiments on ten benchmark datasets demonstrate that DCWNet outperforms state-of-the-art LLIE methods.

---

## 🔥 News

- **[2026/05]** Initial repository and README released.
- **[Coming Soon]** Code, pretrained models, and demo will be released.

---

## 📖 Method Overview

 **Overview of DCWNet:**
<p align="center">
  <img src="figs/network.png" width="900" alt="DCWNet Architecture">
</p>

- **HV-branch for color restoration**
  - Bright Codebook for reliable normal-light color prototypes
  - Dark Codebook for severely underexposed regions
  - Context-Aware Module for spatially consistent color reconstruction
  - Uncertainty-guided fusion for high-entropy regions

- **I-branch for illumination enhancement**
  - Frequency-Decoupled Wavelet Transform (FDWT)
  - Cross-Frequency Interaction Guided Enhancement (CFIGE)
  - Adaptive Illumination-Aware Low-Frequency Optimizer (AILO)

---

## 🛠️ Requirements

The environment is provided in `dcwnet.yaml`.

Main dependencies:

- Python 3.9
- PyTorch 1.13.1
- Torchvision 0.14.1

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

You can refer to the following links to download the datasets.

- [LOLv1](https://daooshee.github.io/BMVC2018website/)
- LOLv2: [Baidu Pan](https://pan.baidu.com/s/17KTa-6GUUW22Q49D5DhhWw?pwd=yixu) (code: `yixu`) and  [One Drive](https://1drv.ms/u/c/2985db836826d183/EYPRJmiD24UggCmCAQAAAAABEbg62rx0FG21FwLQq0jzLg?e=Im12UA) (code: `yixu`) 
- DICM, LIME, MEF, NPE, VV: [Baidu Pan](https://pan.baidu.com/s/1FZ5HWT30eghGuaAqqpJGaw?pwd=yixu)(code: `yixu`) and [One Drive](https://1drv.ms/f/s!AoPRJmiD24UphBNGBbsDmSwppNPf?e=2yGImv)(code: `yixu`)
- SICE: [Baidu Pan](https://pan.baidu.com/s/13ghnpTBfDli3mAzE3vnwHg?pwd=yixu)(code: `yixu`) and [One Drive](https://1drv.ms/u/s!AoPRJmiD24UphAlaTIekdMLwLZnA?e=WxrfOa)(code: `yixu`)
- Sony-Total-Dark(SID): [Baidu Pan](https://pan.baidu.com/s/1mpbwVscbAfQJtkrrzBzJng?pwd=yixu)(code: `yixu`) and [One Drive](https://1drv.ms/u/s!AoPRJmiD24UphAie9l0DuMN20PB7?e=Zc5DcA)(code: `yixu`)

### Recommended Directory Structure
<details close> <summary>datasets (click to expand)</summary>

```
├── datasets
	├── DICM
	├── LIME
	├── LOLdataset
		├── our485
			├──low
			├──high
		├── eval15
			├──low
			├──high
	├── LOLv2
		├── Real_captured
			├── Train
				├── Low
				├── Normal
			├── Test
				├── Low
				├── Normal
		├── Synthetic
			├── Train
				├── Low
				├── Normal
			├── Test
				├── Low
				├── Normal
	├── MEF
	├── NPE
	├── SICE
		├── Dataset
			├── eval
				├── target
				├── test
			├── label
			├── train
				├── 1
				├── 2
				...
		├── SICE_Grad
		├── SICE_Mix
		├── SICE_Reshape
	├── Sony_total_dark
		├── eval
			├── long
			├── short
		├── test
			├── long
				├── 10003
				├── 10006
				...
			├── short
				├── 10003
				├── 10006
				...
		├── train
			├── long
				├── 00001
				├── 00002
				...
			├── short
				├── 00001
				├── 00002
				...
	├── VV
```
</details>

## 🚀 Getting Started


### 2. Testing

Download our weights from [[Google Drive](https://drive.google.com/drive/folders/1Qvayx4F8DoFft5FC9q3-iZYH74-ctJgl?usp=drive_link)]

- **You can test DCWNet as follows, all the results will be saved in the `./output` folder:**

<details close> <summary>(click to expand)</summary>

```bash
# LOLv1
python eval.py --lol

# LOLv2-real
python eval.py --lolv2_real

# LOLv2-syn
python eval.py --lolv2_syn

# SICE
python eval.py --sice

# Sony-Total-Dark
python eval_SID.py

# five unpaired datasets DICM, LIME, MEF, NPE, VV. 
# You can change "--DICM" to the other unpaired datasets "LIME, MEF, NPE, VV".
python eval_unpaired.py
```

</details>

### 2. Training

The training code will be uploaded soon.

## ✨ Qualitative Results

<details>
<summary><strong>LOLv1 Results</strong></summary>
<br>
<p align="center">
  <img src="figs/LOLv1.png" width="900" alt="Qualitative results on LOLv1">
</p>
</details>

<details>
<summary><strong>LOLv2 Results</strong></summary>
<br>
<p align="center">
  <img src="figs/LOLv2r_s.png" width="900" alt="Qualitative results on LOLv2">
</p>
</details>

<details>
<summary><strong>Unpaired Dataset Results</strong></summary>
<br>
<p align="center">
  <img src="figs/unpair.png" width="900" alt="Qualitative results on unpaired datasets">
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


## 💖 Acknowledgement

This repository is built upon the low-light image enhancement community and related HVI/color-space, vector-quantization, and wavelet-based restoration works. We sincerely thank the authors of the benchmark datasets and open-source implementations.

---

## 🔆 Contact

For questions, please open an issue or contact:

- Kangbiao Shi: 18334840904@163.com
