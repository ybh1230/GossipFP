#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='732_semi'
ROOT=../../../..
method='train_gossip_pascal' 


mkdir -p log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
GPUS=${1:-1}
PORT=${2:-29500}

torchrun \
    --nproc_per_node=$GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$PORT \
    $ROOT/$method.py --config=config.yaml --seed 2 --port $PORT 2>&1 | tee log/$now\_$method.txt
