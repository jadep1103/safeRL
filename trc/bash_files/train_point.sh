#!/bin/bash
python main_multi.py \
    --env_name SafetyPointGoal1-v0 \
    --n_envs 4 \
    --device cpu \
    --name TRC_SimplePointMULTI \
    --total_steps 4000000 \
    --n_steps 1024 \
    --wandb \
    --seed 42
