import gymnasium as gym
import torch
import swig

# Affichage de la version de PyTorch et si un GPU est disponible
print("PyTorch version :", torch.__version__)
print("GPU dispo :", torch.cuda.is_available())

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()