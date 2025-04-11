import numpy as np
import safety_gymnasium as safety_gym
import gymnasium as gym
import re

class GymEnv(gym.Env):
    def __init__(self, env_name, seed, max_episode_length, action_repeat):
        self.env_name = env_name
        self._env = gym.make(env_name)
        self._env.reset(seed=seed)
        _, self.robot_name, self.task_name = re.findall('[A-Z][a-z]+', env_name)
        self.robot_name = self.robot_name.lower()
        self.task_name = self.task_name.lower()

        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.goal_threshold = np.inf
        self.hazard_size = 0.2

        self.obs_dim = 2 + 1 + 2 + 2 + 1 + 16
        self.observation_space = gym.spaces.Box(-np.ones(self.obs_dim), np.ones(self.obs_dim))
        self.action_space = self._env.action_space

    def getState(self,obs):
        # Récupération des observations à partir du dictionnaire "obs"
        goal_dir = obs['compass']  # Observation de la direction du but
        goal_dist = np.linalg.norm(obs['goal'])  # Distance au but
        acc = obs['accelerometer'][:2]  # Accélération (composants x et y)
        vel = obs['vel'][:2]  # Vitesse (composants x et y)
        rot_vel = obs['gyro'][2:]  # Vitesse angulaire (composant z)
        lidar = obs['lidar'][:16]  # Lidar, prendre les 16 premières mesures
        state = np.concatenate([goal_dir/0.7, (goal_dist - 1.5)/0.6, acc/8.0, vel/0.2, rot_vel/2.0, (lidar - 0.3)/0.3], axis=0)
        return state

    def getCost(self, h_dist):
        h_coeff = 10.0
        cost = 1.0/(1.0 + np.exp(h_dist*h_coeff))
        return cost

    def getHazardDist(self):
        robot_pos = np.array(self._env.world.robot_pos())
        min_dist = np.inf
        for hazard_pos in self._env.hazards_pos:
            dist = np.linalg.norm(hazard_pos[:2] - robot_pos[:2])
            if dist < min_dist:
                min_dist = dist
        h_dist = min_dist - self.hazard_size
        return h_dist
        
    def reset(self):
        self.t = 0
        obs,info = self._env.reset()
        state = self.getState(obs)
        return state

    def step(self, action):
        reward = 0
        is_goal_met = False
        num_cv = 0

        for _ in range(self.action_repeat):
            obs, r_t, terminated, truncated, info = self._env.step(action)
            if info['cost'] > 0:
                num_cv += 1
            try:
                if info['goal_met']:
                    is_goal_met = True
            except:
                pass
            reward += r_t
            self.t += 1
            done = terminated or truncated or self.t == self.max_episode_length
            if done:
                break

        state = self.getState(obs)
        h_dist = self.getHazardDist()

        info['goal_met'] = is_goal_met
        info['cost'] = self.getCost(h_dist)
        info['num_cv'] = num_cv
        return state, reward, done, info

    def render(self, **args):
        return self._env.render(**args)

    def close(self):
        self._env.close()


def Env(env_name, seed, max_episode_length=1000, action_repeat=1):
    env_list = ['Jackal-v0', 'Doggo-v0', 'Safexp-PointGoal1-v0', 'Safexp-CarGoal1-v0','SafetyPointGoal1-v0']
    if not env_name in env_list:
        raise ValueError(f'Invalid environment name.\nSupport Env list: {env_list}') 
    if 'jackal' in env_name.lower() or 'doggo' in env_name.lower():
        return gym.make(env_name)
    else:
        return GymEnv(env_name, seed, max_episode_length, action_repeat)
