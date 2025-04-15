import gymnasium as gym
import safety_gymnasium
import wandb
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from wandb.integration.sb3 import WandbCallback

# Wrapper pour corriger le retour de step()
class SafetyGymWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        print(f"Reward: {reward}, Cost: {cost}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        info["cost"] = cost  # Ajout du coût dans info
        return obs, reward, terminated, truncated, info

def make_env(env_id):
    def _init():
        env = SafetyGymWrapper(safety_gymnasium.make(env_id))
        return env
    return _init


def train(env_id, total_timesteps=1_000_000, num_envs=4):
    run_name = f"PPO-{env_id}-VecNorm"

    wandb.init(
        project="safe-rl",
        name=run_name,
        config={
            "env": env_id,
            "algo": "PPO",
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
        },
        monitor_gym=False,
        save_code=True,
    )

    # Création des environnements vectorisés avec wrapper
    vec_env = SubprocVecEnv([make_env(env_id) for _ in range(num_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    vec_env = VecMonitor(vec_env)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=f"./ppo_logs/{env_id}/"
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"./models/{env_id}/",
            verbose=2,
        )
    )

    os.makedirs(f"./models/{env_id}/", exist_ok=True)
    model.save(f"./models/{env_id}/ppo_model")
    vec_env.save(f"./models/{env_id}/vec_normalize.pkl")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="SafetyCarGoal1-v0",
                        choices=["SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyDoggoGoal1-v0"])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_envs", type=int, default=4)
    args = parser.parse_args()

    train(env_id=args.env, total_timesteps=args.timesteps, num_envs=args.num_envs)
