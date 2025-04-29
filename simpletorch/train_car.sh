#!/bin/bash
python train_trcsimple.py \
    --env_name SafetyCarGoal1-v0 \
    --n_envs 1 \
    --device cpu \
    --name TRC_SimpleCar \
    --total_steps 1000000 \
    --n_steps 4000 \
    --wandb