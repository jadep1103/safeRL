#!/bin/bash
python main.py \
    --env_name SafetyPointGoal2-v0 \
    --n_envs 1 \
    --device gpu \
    --name TRC_FINAL_point\
    --test \
    --seed 42 \
    --lr 1e-4 \
    --hidden_dim 256 \
    --gae_coeff 0.95 \
    --ent_coeff 0.01 \
    --max_kl 0.005 \
    --cost_d 0.01 \
    --cost_alpha 0.2 \
    --line_decay 0.8 \
    --num_conjugate 10 \
    --damping_coeff 0.01 \
    --activation ReLU \
    --wandb
