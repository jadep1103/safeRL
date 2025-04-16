#!/bin/bash
python train_trcsimple.py \
    --env_name SafetyPointGoal0-v0 \
    --n_envs 1 \
    --device cpu \
    --name TRC_SimplePoint \
    --total_steps 10000 \
    --n_steps 1000
