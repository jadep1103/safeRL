#!/bin/bash
python train_trcsimple.py \
    --env_name SafetyPointGoal1-v0 \
    --n_envs 1 \
    --device cpu \
    --name TRC_SimplePoint \
    --total_steps 300000 \
    --n_steps 4000 \
    --wandb