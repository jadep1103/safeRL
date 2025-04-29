# ===== add python path ===== #
import glob
import sys
import os
PATH = os.getcwd()
for dir_idx, dir_name in enumerate(PATH.split('/')):
    dir_path = '/'.join(PATH.split('/')[:(dir_idx+1)])
    file_list = [os.path.basename(sub_dir) for sub_dir in glob.glob(f"{dir_path}/.*")]
    if '.trc_package' in file_list:
        PATH = dir_path
        break
if not PATH in sys.path:
    sys.path.append(PATH)
# =========================== #

# Imports
import gymnasium as gym
import safety_gymnasium
import safety_gymnasium.vector  # IMPORTANT !
from logger import Logger
from agent import Agent
from datetime import datetime
import numpy as np
import argparse
import random
import wandb
import torch
import time

def getPaser():
    parser = argparse.ArgumentParser(description='TRC')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--name', type=str, default='TRC')
    parser.add_argument('--save_freq', type=int, default=int(1e6))
    parser.add_argument('--total_steps', type=int, default=int(1e7))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='SafetyPointGoal1-v0')
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=4000)
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--log_std_init', type=float, default=-1.0)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--gae_coeff', type=float, default=0.97)
    parser.add_argument('--ent_coeff', type=float, default=0.0)
    parser.add_argument('--damping_coeff', type=float, default=0.01)
    parser.add_argument('--num_conjugate', type=int, default=10)
    parser.add_argument('--line_decay', type=float, default=0.8)
    parser.add_argument('--max_kl', type=float, default=0.001)
    parser.add_argument('--cost_d', type=float, default=25.0/1000.0)
    parser.add_argument('--cost_alpha', type=float, default=0.125)
    return parser

def train(args):
    print("[DEBUG] Starting training...")

    if args.wandb:
        project_name = '[TRC_torch] safety_gym'
        wandb.init(project=project_name, config=args)
        run_idx = wandb.run.name.split('-')[-1]
        wandb.run.name = f"{args.name}-{run_idx}"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    vec_env = safety_gymnasium.vector.make(
        args.env_name,
        num_envs=args.n_envs,
        render_mode=None,
    )

    args.obs_dim = vec_env.single_observation_space.shape[0]
    args.action_dim = vec_env.single_action_space.shape[0]
    args.action_bound_min = vec_env.single_action_space.low
    args.action_bound_max = vec_env.single_action_space.high

    agent = Agent(args)

    # loggers
    objective_logger = Logger(args.save_dir, 'objective')
    cost_surrogate_logger = Logger(args.save_dir, 'cost_surrogate')
    v_loss_logger = Logger(args.save_dir, 'v_loss')
    cost_v_loss_logger = Logger(args.save_dir, 'cost_v_loss')
    cost_var_v_loss_logger = Logger(args.save_dir, 'cost_var_v_loss')
    kl_logger = Logger(args.save_dir, 'kl')
    entropy_logger = Logger(args.save_dir, 'entropy')
    score_logger = Logger(args.save_dir, 'score')
    eplen_logger = Logger(args.save_dir, 'eplen')
    cost_logger = Logger(args.save_dir, 'cost')
    cv_logger = Logger(args.save_dir, 'cv')

    observations, infos = vec_env.reset(seed=args.seed)
    reward_history = [[] for _ in range(args.n_envs)]
    cost_history = [[] for _ in range(args.n_envs)]
    cv_history = [[] for _ in range(args.n_envs)]
    env_cnts = np.zeros(args.n_envs)
    total_step = 0
    save_step = 0

    while total_step < args.total_steps:
        trajectories = [[] for _ in range(args.n_envs)]
        step = 0

        while step < args.n_steps:
            env_cnts += 1
            step += args.n_envs
            total_step += args.n_envs

            with torch.no_grad():
                obs_tensor = torch.tensor(observations, device=args.device, dtype=torch.float32)
                action_tensor, clipped_action_tensor = agent.getAction(obs_tensor, True)
                clipped_actions = clipped_action_tensor.cpu().numpy()

            next_obs, rewards, costs, terminations, truncations, infos = vec_env.step(clipped_actions)
            dones = np.logical_or(terminations, truncations)

            for i in range(args.n_envs):
                reward_history[i].append(rewards[i])
                cost_history[i].append(costs[i])

                if isinstance(infos, list):
                    info = infos[i]
                else:
                    info = infos

                cv_history[i].append(info.get('num_cv', 0))

                fail = env_cnts[i] < args.max_episode_steps if dones[i] else False
                terminal_obs = info.get('terminal_observation', next_obs[i]) if dones[i] else next_obs[i]

                trajectories[i].append([observations[i], clipped_actions[i], rewards[i], costs[i], dones[i], fail, terminal_obs])

                if dones[i]:
                    ep_len = len(reward_history[i])
                    score = np.sum(reward_history[i])
                    ep_cv = np.sum(cv_history[i])
                    cost_sum = np.sum(cost_history[i])

                    score_logger.write([ep_len, score])
                    eplen_logger.write([ep_len, ep_len])
                    cost_logger.write([ep_len, cost_sum])
                    cv_logger.write([ep_len, ep_cv])

                    reward_history[i].clear()
                    cost_history[i].clear()
                    cv_history[i].clear()
                    env_cnts[i] = 0

            observations = next_obs

        v_loss, cost_v_loss, cost_var_v_loss, objective, cost_surrogate, kl, entropy, optim_case = agent.train(trajectories)

        log_data = {
            "rollout/score": score_logger.get_avg(5),
            "rollout/ep_len": eplen_logger.get_avg(5),
            "rollout/ep_cv": cv_logger.get_avg(5),
            "rollout/cost_sum_mean": cost_logger.get_avg(5),
            "rollout/cost_sum_cvar": cost_logger.get_cvar(agent.sigma_unit, 5),
            "train/value_loss": v_loss_logger.get_avg(),
            "train/cost_value_loss": cost_v_loss_logger.get_avg(),
            "train/cost_var_value_loss": cost_var_v_loss_logger.get_avg(),
            "metric/objective": objective_logger.get_avg(),
            "metric/cost_surrogate": cost_surrogate_logger.get_avg(),
            "metric/kl": kl_logger.get_avg(),
            "metric/entropy": entropy_logger.get_avg(),
        }

        if args.wandb:
            wandb.log(log_data)

        if total_step - save_step >= args.save_freq:
            save_step += args.save_freq
            agent.save()
            objective_logger.save()
            cost_surrogate_logger.save()
            v_loss_logger.save()
            cost_v_loss_logger.save()
            cost_var_v_loss_logger.save()
            entropy_logger.save()
            kl_logger.save()
            score_logger.save()
            eplen_logger.save()
            cv_logger.save()
            cost_logger.save()

def test(args):
    pass

if __name__ == "__main__":
    parser = getPaser()
    args = parser.parse_args()
    args.save_dir = f"results/{args.name}_s{args.seed}"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_idx}"

    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')
    args.device = device

    if args.test:
        test(args)
    else:
        train(args)
