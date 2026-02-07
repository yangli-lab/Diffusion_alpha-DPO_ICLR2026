#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
DATASET=${1:-"pick2pic_v2_test_100"}
export MASTER_PORT=29501

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT ./scripts/inference.py $DATASET