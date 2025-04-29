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


from logger import Logger
from agent import Agent
from datetime import datetime
import safety_gymnasium

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import argparse
import random
import wandb
import torch
import time

def getPaser():
    parser = argparse.ArgumentParser(description='TRC')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--slack',  action='store_true', help='use slack?')
    parser.add_argument('--test',  action='store_true', help='test or train?')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--name', type=str, default='TRC', help='save name.')
    parser.add_argument('--save_freq', type=int, default=int(1e6), help='# of time steps for save.')
    parser.add_argument('--slack_freq', type=int, default=int(2.5e6), help='# of time steps for slack message.')
    parser.add_argument('--total_steps', type=int, default=int(1e7), help='total training steps.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    # for env
    parser.add_argument('--env_name', type=str, default='SafetyPointGoal1-v0', help='gym environment name.')
    parser.add_argument('--max_episode_steps', type=int, default=1000, help='# of maximum episode steps.')
    parser.add_argument('--n_envs', type=int, default=1, help='gym environment name.')
    parser.add_argument('--n_steps', type=int, default=4000, help='update after collecting n_steps.')
    # for networks
    parser.add_argument('--activation', type=str, default='ReLU', help='activation function. ReLU, Tanh, Sigmoid...')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the number of hidden layer\'s node.')
    parser.add_argument('--log_std_init', type=float, default=-1.0, help='log of initial std.')
    # for RL
    parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor.')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate.')
    parser.add_argument('--n_epochs', type=int, default=200, help='update epochs.')
    parser.add_argument('--gae_coeff', type=float, default=0.97, help='gae coefficient.')
    parser.add_argument('--ent_coeff', type=float, default=0.0, help='gae coefficient.')
    # trust region
    parser.add_argument('--damping_coeff', type=float, default=0.01, help='damping coefficient.')
    parser.add_argument('--num_conjugate', type=int, default=10, help='# of maximum conjugate step.')
    parser.add_argument('--line_decay', type=float, default=0.8, help='line decay.')
    parser.add_argument('--max_kl', type=float, default=0.001, help='maximum kl divergence.')
    # constraint
    parser.add_argument('--cost_d', type=float, default=25.0/1000.0, help='constraint limit value.')
    parser.add_argument('--cost_alpha', type=float, default=0.125, help='CVaR\'s alpha.')
    return parser
def train(args):
    print("[DEBUG] Starting training...")

    # wandb
    if args.wandb:
        project_name = '[TRC_torch] safety_gym'
        wandb.init(project=project_name, config=args)
        run_idx = wandb.run.name.split('-')[-1]
        wandb.run.name = f"{args.name}-{run_idx}"
        print(f"[DEBUG] wandb initialized: {wandb.run.name}")


    # for random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[DEBUG] Seeds set.")

    # define Environment
    if args.n_envs == 1:
        vec_env = safety_gymnasium.make(args.env_name)
    else:
        def make_env():
            def _init():
                return safety_gymnasium.make(args.env_name)
            return _init
        vec_env = make_vec_env(
            env_id=make_env(), n_envs=args.n_envs, seed=args.seed,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={'start_method': 'spawn'},
        )
    print(f"[DEBUG] Environment {args.env_name} created.")

    # set args value for env
    args.obs_dim = vec_env.observation_space.shape[0]
    args.action_dim = vec_env.action_space.shape[0]
    args.action_bound_min = vec_env.action_space.low
    args.action_bound_max = vec_env.action_space.high
    print(f"[DEBUG] obs_dim: {args.obs_dim}, action_dim: {args.action_dim}")

    # define agent
    agent = Agent(args)
    print("[DEBUG] Agent initialized.")

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
    print("[DEBUG] Loggers initialized.")

    # training loop
    observations = vec_env.reset(seed=args.seed)
    reward_history = [[] for _ in range(args.n_envs)]
    cost_history = [[] for _ in range(args.n_envs)]
    cv_history = [[] for _ in range(args.n_envs)]
    env_cnts = np.zeros(args.n_envs)
    total_step = 0
    slack_step = 0
    save_step = 0

    while total_step < args.total_steps:
        print(f"[DEBUG] === New outer loop | Total step: {total_step} ===")
        trajectories = [[] for _ in range(args.n_envs)]
        step = 0

        while step < args.n_steps:
            #print(f"[DEBUG] Step {total_step} | Collecting transitions")
            env_cnts += 1
            step += args.n_envs
            total_step += args.n_envs

            with torch.no_grad():
                obs, _ = observations
                obs_tensor = torch.tensor(obs, device=args.device, dtype=torch.float32).unsqueeze(0)
                #print(f"[DEBUG] obs_tensor shape: {obs_tensor.shape}")

                action_tensor, clipped_action_tensor = agent.getAction(obs_tensor, True)
                actions = action_tensor.cpu().numpy()
                clipped_actions = clipped_action_tensor.cpu().numpy()
                #print(f"[DEBUG] Action taken: {clipped_actions}")

            next_obs, reward, cost, terminated, truncated, info = vec_env.step(clipped_actions.squeeze(0))
            if total_step < 200000:
                reward *= 20
            done = terminated or truncated
            next_observations = (next_obs, info)

            # ===== DEBUG PATCH =====
            if total_step % 5000 == 0:
                print("\n[DEBUG] ====== Step", total_step, "======")
                print("[DEBUG] obs (first 5 dims):", obs[:5] if isinstance(obs, np.ndarray) else np.array(obs)[:5])
                print("[DEBUG] reward:", reward)
                print("[DEBUG] cost:", cost)
                if isinstance(info, dict):
                    if 'goal_distance' in info:
                        print("[DEBUG] goal_distance:", info['goal_distance'])
                    if 'cost' in info:
                        print("[DEBUG] info cost field:", info['cost'])
                    if 'num_cv' in info:
                        print("[DEBUG] constraint violations (num_cv):", info['num_cv'])
            # ========================


            reward_history[0].append(reward)
            cv_history[0].append(info.get('num_cv', 0))
            cost_history[0].append(cost)

            fail = env_cnts[0] < args.max_episode_steps if done else False
            terminal_obs = info.get('terminal_observation', next_obs) if done else next_obs
            trajectories[0].append([obs, actions[0], reward, cost, done, fail, terminal_obs])

            if done:
                ep_len = len(reward_history[0])
                score = np.sum(reward_history[0])
                ep_cv = np.sum(cv_history[0])
                cost_sum = np.sum(cost_history[0])

                score_logger.write([ep_len, score])
                eplen_logger.write([ep_len, ep_len])
                cost_logger.write([ep_len, cost_sum])
                cv_logger.write([ep_len, ep_cv])

                reward_history[0].clear()
                cost_history[0].clear()
                cv_history[0].clear()
                env_cnts[0] = 0
                observations = vec_env.reset(seed=args.seed)
            else:
                observations = next_observations

        print("[DEBUG] Updating agent...")
        v_loss, cost_v_loss, cost_var_v_loss, objective, cost_surrogate, kl, entropy, optim_case = agent.train(trajectories)
        optim_hist = np.histogram([optim_case], bins=np.arange(0, 6))

        objective_logger.write([step, objective])
        cost_surrogate_logger.write([step, cost_surrogate])
        v_loss_logger.write([step, v_loss])
        cost_v_loss_logger.write([step, cost_v_loss])
        cost_var_v_loss_logger.write([step, cost_var_v_loss])
        kl_logger.write([step, kl])
        entropy_logger.write([step, entropy])

        log_data = {
            "rollout/score": score_logger.get_avg(5), 
            "rollout/ep_len": eplen_logger.get_avg(5),
            "rollout/ep_cv": cv_logger.get_avg(5),
            "rollout/cost_sum_mean": cost_logger.get_avg(5),
            "rollout/cost_sum_cvar": cost_logger.get_cvar(agent.sigma_unit, 5),
            "train/value_loss":v_loss_logger.get_avg(), 
            "train/cost_value_loss":cost_v_loss_logger.get_avg(), 
            "train/cost_var_value_loss":cost_var_v_loss_logger.get_avg(), 
            "metric/objective":objective_logger.get_avg(), 
            "metric/cost_surrogate":cost_surrogate_logger.get_avg(), 
            "metric/kl":kl_logger.get_avg(), 
            "metric/entropy":entropy_logger.get_avg(),
            "metric/optim_case":wandb.Histogram(np_histogram=optim_hist), 
        }

        print(f"[DEBUG] log_data: {log_data}")
        print("le caca est cuit")
        print("[manual save] Forcing final save...")
        agent.save()
        print("[DEBUG] Forcing final save of all logs...")
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
        print("[DEBUG] All logs saved successfully.")


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
    # define Environment
    #env = safety_gymnasium.make(args.env_name, render_mode='human')

    #take offline video
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_path = f"videos/{args.name}_{args.env_name}_{now}"
    os.makedirs(video_path, exist_ok=True)

    env = safety_gymnasium.make(args.env_name, render_mode="human",max_episode_steps=1000)


    obs, info = env.reset(seed=args.seed)
    # set args value for env
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.action_bound_min = env.action_space.low
    args.action_bound_max = env.action_space.high

    # define agent
    agent = Agent(args)

    scores = []
    cvs = []

    epochs = 100
    for epoch in range(epochs):
        state, _ = env.reset(seed = args.seed)
        done = False
        score = 0
        cv = 0
        step = 0

        while True:
            step += 1
            with torch.no_grad():
                obs_tensor = torch.from_numpy(np.array(state)).float().to(args.device)
                #obs_tensor = torch.tensor(state, device=args.device, dtype=torch.float32)
                action_tensor, clipped_action_tensor = agent.getAction(obs_tensor, False)
                action = action_tensor.detach().cpu().numpy()
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
            next_state, reward, cost, terminated, truncated, info = env.step(clipped_action)
            env.render()

            state = next_state
            print("Terminated",terminated)
            print("Truncated",truncated)
            done = terminated or truncated
            score += reward
            cv += info.get('num_cv', 0)
            #cv += info['num_cv']
            print("caca",done)
            if done or step >= args.max_episode_steps:
                break

                

        print(f"[TEST] Score: {score:.3f} | Constraint Violations: {cv}")
        scores.append(score)
        cvs.append(cv)
        print(score, cv)

    print(f"[TEST] Moyenne score: {np.mean(scores):.3f}, Moyenne CV: {np.mean(cvs)}")
    env.close()


if __name__ == "__main__":
    parser = getPaser()
    args = parser.parse_args()
    # ==== processing args ==== #
    # save_dir
    args.save_dir = f"results/{args.name}_s{args.seed}"
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_idx}"
    # device
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device('cuda:0')
        print('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        print('[torch] cpu is used.')
    args.device = device
    # ========================= #

    if args.test:
        test(args)
    else:
        train(args)
