import gymnasium as gym
import safety_gymnasium
import wandb
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from wrapper import SafetyGymCompatibilityWrapper
class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if "episode" in self.locals:
            ep_info = self.locals["episode"]
            if isinstance(ep_info, dict):
                wandb.log({
                    "score_log": ep_info.get("r", 0.0),
                    "cost_log": ep_info.get("cost", 0.0),
                    "cv_log": ep_info.get("cost", 0.0) / (ep_info.get("l", 1) + 1e-8),  # coût moyen par étape
                    "episode_length": ep_info.get("l", 0),
                    "timesteps": self.num_timesteps,
                })
        return True


def make_env(env_id):
    def _init():
        env = safety_gymnasium.make(env_id)
        env = SafetyGymCompatibilityWrapper(env)
        return env
    return _init


def train(env_id, total_timesteps=1000000, num_envs=4):
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
        tags=["train"],
        group=f"Training-{env_id}",
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
        callback=CustomWandbCallback()
        )

    os.makedirs(f"./models/{env_id}/", exist_ok=True)
    model.save(f"./models/{env_id}/ppo_model")
    vec_env.save(f"./models/{env_id}/vec_normalize.pkl")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="SafetyCarGoal1-v0",
                        choices=["SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyDoggoGoal1-v0","SafetyPointGoal2-v0","SafetyCarGoal2-v0"])
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--num_envs", type=int, default=4)
    args = parser.parse_args()

    # Test avec un seul environnement pour déboguer si nécessaire
    # train(env_id=args.env, total_timesteps=args.timesteps, num_envs=1)
    train(env_id=args.env, total_timesteps=args.timesteps, num_envs=args.num_envs)
