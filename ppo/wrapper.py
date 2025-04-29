import gymnasium as gym

class SafetyGymCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper pour rendre Safety-Gymnasium compatible avec Stable-Baselines3 :
    - step() retourne obs, reward, done, info (fusion de terminated et truncated)
    - reset() retourne (obs, info)
    """

    def __init__(self, env):
        super(SafetyGymCompatibilityWrapper, self).__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        
        if not isinstance(info, dict):
            info = {}
        info["cost"] = cost
        info["TimeLimit.truncated"] = truncated and not terminated

        # 🚨 Retourne bien 5 éléments !
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
