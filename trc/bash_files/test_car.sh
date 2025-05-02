#!/bin/bash
python main.py \
    --env_name SafetyCarGoal1-v0 \
    --n_envs 1 \
    --device cpu \
    --name TRC_SimpleCarSINGLEBOOSTED\
    --seed 42 \
    --wandb \
    --test
