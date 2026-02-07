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
export MODEL_NAME="Model_Name"
export VAE="sdxl-vae-fp16-fix"
export DATASET=/path/to/dataset
export IP_ADDR=127.0.0.1
export PORT_ADDR=7890

noise_portion=0.3

SEED=2025
CHECKPOINT=/path/to/checkpoint

accelerate launch \
  --config_file ./launchers/config.yaml \
  --main_process_ip $IP_ADDR \
  --main_process_port $PORT_ADDR \
  --num_machines 1 \
  --machine_rank 0 \
  --num_processes 8 \
  /scripts/main.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=${DATASET}/metadata.jsonl \
  --train_data_dir=${DATASET} \
  --train_batch_size=4 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=64 \
  --max_train_steps=600 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=8.192e-9 --scale_lr \
  --checkpointing_steps 200 \
  --beta_dpo 4000 \
  --alpha_dpo 0.9999 \
  --sdxl \
  --reweight_step -1 \
  --output_dir=${CHECKPOINT}/checkpoint_savedir \
  --seed ${SEED} 2>&1 | tee logs/dpo_official/training.log
```

## 🔍 Inference

After training, you can generate samples using the inference script:

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
DATASET=${1:-"pick2pic_v2_test_100"}
export MASTER_PORT=29501

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT ./scripts/inference.py $DATASET
```

## 📚 Dependencies
This project builds upon the following open-source repository:
- [Diffusion-DPO](https://github.com/SalesforceAIResearch/DiffusionDPO) — A foundational codebase for DPO in diffusion models.
- [Diffusion-SPO](https://github.com/RockeyCoss/SPO) — Official implementation of paper: [Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](https://arxiv.org/abs/2406.04314)

## 📖 Citation

If you find this work useful in your research, please consider citing our paper:
```bibtex
@article{author2025alpha,
  title={α-DPO: Robust Preference Alignment for Diffusion Models via α Divergence},
  author={Yang Li, Songlin Yang, Wei Wang, Xiaoxuan Han, Jing Dong},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
Thank you for your interest in Alpha-DPO! 🙌
