#!/bin/bash
python train_trcsimple.py \
    --env_name SafetyCarGoal1-v0 \
    --n_envs 1 \
    --device cpu \
    --name TRC_SimpleCar \
    --test
