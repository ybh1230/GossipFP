#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='732_semi'
ROOT=../../../..
method='train_gossip_pascal' 


mkdir -p log

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

python $ROOT/$method.py --config=config.yaml --seed 2 2>&1 | tee log/$now\_$method.txt
