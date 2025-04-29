import os
import gymnasium as gym
import argparse
import wandb
import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wrapper import SafetyGymCompatibilityWrapper


# =========================
def test(env_id, n_episodes=5):
    model_path = f"./models/{env_id}/ppo_model.zip"
    vecnorm_path = f"./models/{env_id}/vec_normalize.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vecnorm_path):
        print(f"Modèle ou normalisation manquante pour {env_id}")
        return

    print(f"Chargement du modèle et normalisation pour {env_id}...")

    env = env = SafetyGymCompatibilityWrapper(safety_gymnasium.make(env_id, render_mode="human"))   # Utilisation du wrapper custom
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    wandb.init(project="safe-rl", name=f"test_{env_id}", monitor_gym=False, save_code=True,tags=["test"],group=f"Testing-{env_id}")

    rewards_list = []
    lengths_list = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        total_cost = 0.0
        length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]
            if isinstance(info[0], dict) and "cost" in info[0]:
                total_cost += info[0]["cost"]  # cost dans info[0] car DummyVecEnv

            done = done[0]
            length += 1

        # Eviter division par 0
        cost_per_step = total_cost / (length + 1e-8)

        print(f"Épisode {episode+1} terminé – Reward: {total_reward:.2f}, Cost: {total_cost:.2f}")

        wandb.log({
            "episode": episode + 1,
            "score_log": total_reward,
            "cost_log": total_cost,
            "cv_log": cost_per_step,
            "episode_length": length
        })

        rewards_list.append(total_reward)
        lengths_list.append(length)

    print(f"\nStatistiques sur {n_episodes} épisodes :")
    print(f"Reward moyen : {sum(rewards_list)/n_episodes:.2f}")
    print(f"Longueur moyenne : {sum(lengths_list)/n_episodes:.2f}")

    wandb.finish()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="SafetyPointGoal1-v0",
                        choices=["SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyDoggoGoal1-v0","SafetyPointGoal2-v0","SafetyCarGoal2-v0"])
    parser.add_argument("--episodes", type=int, default=5, help="Nombre d'épisodes de test")
    args = parser.parse_args()

    test(env_id=args.env, n_episodes=args.episodes)
