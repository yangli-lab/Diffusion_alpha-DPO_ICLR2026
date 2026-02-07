#!/bin/bash

export MODEL_NAME="Model_Name"
export VAE="sdxl-vae-fp16-fix"
export DATASET=/path/to/dataset

export IP_ADDR=127.0.0.1
export PORT_ADDR=7890

noise_portion=0.3

SEED=2025
# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps

CHECKPOINT=/path/to/checkpoint


accelerate launch \
  --config_file ./launchers/config.yaml \
  --main_process_ip $IP_ADDR \
  --main_process_port $PORT_ADDR \
  --num_machines 1 \
  --machine_rank 0 \
  --num_processes 8 \
  ./scripts/main.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=${DATASET}/metadata.jsonl \
  --train_data_dir=${DATASET} \
  --train_batch_size=4 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=64 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=8.192e-9 --scale_lr \
  --checkpointing_steps 200 \
  --beta_dpo 4000 \
  --alpha_dpo 0.9999 \
  --sdxl \
  --reweight_step -1 \
  --output_dir=${CHECKPOINT}/checkpoint_savedir \
  --seed ${SEED} 2>&1 | tee logs/dpo_official/training.log
