import gymnasium as gym
import safety_gymnasium
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def test(env_id):
    model_path = f"./models/{env_id}/ppo_model.zip"
    vecnorm_path = f"./models/{env_id}/vec_normalize.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vecnorm_path):
        print(f"Modèle ou normalisation manquante pour {env_id}")
        return

    print(f"Chargement du modèle et normalisation pour {env_id}...")

    env = DummyVecEnv([lambda: gym.make(env_id, render_mode="human")])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    done = False
    total_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward[0]
        if done[0]:
            break

    print(f"Épisode terminé – Reward total : {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="SafetyPointGoal1-v0",
                        choices=["SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyDoggoGoal1-v0"])
    args = parser.parse_args()

    test(env_id=args.env)