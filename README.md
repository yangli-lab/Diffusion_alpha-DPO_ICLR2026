# Diffusion_alpha-DPO_ICLR2026

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="Paper">
  </a>
  &nbsp;&nbsp;
  <a href="https://your-project-page-url.com">
    <img src="https://img.shields.io/badge/Project_Page-Website-green" alt="Project Page">
  </a>
</p>

This repository is the official implementation of the ICLR2026 paper: **"α-DPO: Robust Preference Alignment for Diffusion Models via α Divergence"**.

---

## 📌 Method Overview

Our method introduces a novel approach to align diffusion models with human preferences through a refined Direct Preference Optimization (DPO) framework tailored for generative diffusion processes.

![Method Diagram](assets/method_diagram.png)

*Figure: Overview of the Alpha-DPO framework for diffusion models.*

---

## 🛠️ Installation

### Torch Environment

### Conda Environment Setup

We recommend using `conda` to manage dependencies. First, create and activate a new environment:

```bash
conda create -n alpha_dpo python=3.9 -y
conda activate alpha_dpo
```

Then install the required packages from ``requirements.txt``.

```bash
pip install -r requirements.txt
```
**Note**: Make sure you have CUDA-compatible PyTorch installed if you plan to use GPU acceleration.

## ▶️ Training

To start training with your configured settings, run the provided training script:

```bash
#!/bin/bash
python train.py \
  --config configs/alpha_dpo_config.yaml \
  --output_dir ./outputs \
  --gradient_accumulation_steps 4 \
  --mixed_precision fp16
```

## 🔍 Inference

After training, you can generate samples using the inference script:

```bash
#!/bin/bash
python inference.py \
  --model_path ./outputs/final_model \
  --prompt "A photorealistic cat sitting on a windowsill" \
  --num_samples 4 \
  --output_dir ./results
```

## 📚 Dependencies
This project builds upon the following open-source repository:
- [Diffusion-DPO](https://github.com/SalesforceAIResearch/DiffusionDPO) — A foundational codebase for DPO in diffusion models.
- [Diffusion-SPO](https://github.com/RockeyCoss/SPO) — Official implementation of paper: [Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](https://arxiv.org/abs/2406.04314)

## 📖 Citation

If you find this work useful in your research, please consider citing our paper:
```bibtex
@article{author2025alpha,
  title={Alpha-DPO: Diffusion},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
Thank you for your interest in Alpha-DPO! 🙌
