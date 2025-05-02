#!/bin/bash
python main.py \
    --env_name SafetyCarGoal1-v0 \
    --n_envs 1 \
    --device cpu \
    --name TRC_SimpleCarSINGLEBOOSTED \
    --total_steps 1000000 \
    --n_steps 1024 \
    --wandb \
    --seed 42
