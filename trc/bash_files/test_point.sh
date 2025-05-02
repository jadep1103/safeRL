#!/bin/bash
python main.py \
    --env_name SafetyPointGoal1-v0 \
    --n_envs 1 \
    --device cpu \
    --name TRC_SimplePointSINGLE \
    --test \
    --wandb \
    --seed 42 
