import gymnasium as gym
import safety_gymnasium

# Wrapper pour corriger le retour de step()
class SafetyGymWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        print(f"Reward: {reward}, Cost: {cost}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        info["cost"] = cost  # Ajout du coût dans info
        return obs, reward, terminated, truncated, info

env = SafetyGymWrapper(safety_gymnasium.make("SafetyPointGoal1-v0"))

obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert "cost" in info