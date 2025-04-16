import numpy as np
import safety_gymnasium as safety_gym
import gymnasium as gym
import re

class GymEnv(gym.Env):
    def __init__(self, env_name, seed, max_episode_length, action_repeat):
        print(f"[INIT] Creating environment '{env_name}' with seed {seed}")
        self.env_name = env_name
        self._env = safety_gym.make(env_name)
        self._env.reset(seed=seed)
        _, self.robot_name, self.task_name = re.findall('[A-Z][a-z]+', env_name)
        self.robot_name = self.robot_name.lower()
        self.task_name = self.task_name.lower()

        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.goal_threshold = np.inf
        self.hazard_size = 0.2

        obs, _ = self._env.reset(seed=seed)
        state = self.getState(obs)
        self.obs_dim = state.shape[0]
        self.observation_space = gym.spaces.Box(-np.ones(self.obs_dim), np.ones(self.obs_dim))
        self.action_space = self._env.action_space
        print(f"[INIT] Environment initialized with obs_dim = {self.obs_dim}")

    def getState(self, obs):
        print("[STATE] Computing state from observation.")
        if not isinstance(obs, dict):
            print("[STATE] Observation is not a dict, returning raw obs.")
            return obs

        features = []

        if 'compass' in obs:
            features.append(obs['compass'] / 0.7)
        if 'goal' in obs:
            goal_dist = np.linalg.norm(obs['goal'])
            features.append((goal_dist - 1.5) / 0.6)
        if 'accelerometer' in obs:
            features.append(obs['accelerometer'][:2] / 8.0)
        if 'vel' in obs:
            features.append(obs['vel'][:2] / 0.2)
        elif 'velocimeter' in obs:
            features.append(obs['velocimeter'][:2] / 0.2)
        if 'gyro' in obs:
            features.append(obs['gyro'][2:] / 2.0)
        if 'lidar' in obs:
            features.append((obs['lidar'][:16] - 0.3) / 0.3)

        if len(features) == 0:
            raise ValueError(f"[STATE] Aucune observation utilisable trouvée dans obs: {obs}")

        state = np.concatenate(features, axis=0)
        print(f"[STATE] Final state shape: {state.shape}")
        return state

    # def getCost(self, h_dist):
    #     h_coeff = 10.0
    #     cost = 1.0 / (1.0 + np.exp(h_dist * h_coeff))
    #     print(f"[COST] Hazard distance: {h_dist}, cost: {cost}")
    #     return cost

    # def getHazardDist(self):
    #     robot_pos = np.array(self._env.world.robot_pos())
    #     min_dist = np.inf
    #     for hazard_pos in self._env.hazards_pos:
    #         dist = np.linalg.norm(hazard_pos[:2] - robot_pos[:2])
    #         if dist < min_dist:
    #             min_dist = dist
    #     h_dist = min_dist - self.hazard_size
    #     return h_dist

    def reset(self, verbose=False):
        print("[RESET] Resetting environment.")
        self.t = 0
        obs, info = self._env.reset()

        if verbose:
            print("Clés dispo dans obs:", list(obs.keys()))

        state = self.getState(obs)
        return state

    def step(self, action):
        print(f"[STEP] Starting step with action: {action}")
        reward = 0
        is_goal_met = False
        num_cv = 0
        terminated = False 
        truncated = False 

        for i in range(self.action_repeat):
            print(f"[STEP] Repeat {i+1}/{self.action_repeat}")
            result = self._env.step(action)
            print(f"[DEBUG] step() result type: {type(result)}, length: {len(result)}")
            for i, r in enumerate(result):
                print(f"  -> Element {i}: {type(r)}")
            print(f"[STEP] Raw result from env: {result}, len={len(result)}")
            obs, r_t, cost, term, trunc, info = result
            terminated = terminated or term
            truncated = truncated or trunc

            if 'cost' not in info: 
                info['cost'] = cost 

            if info.get('cost', 0) > 0:
                num_cv += 1
            if info.get('goal_met', False):
                is_goal_met = True

            reward += r_t
            self.t += 1
            if terminated or truncated or self.t == self.max_episode_length:
                break

        state = self.getState(obs)

        info['goal_met'] = is_goal_met
        info['num_cv'] = num_cv
        info['cost_sum'] = info.get('cost_sum', 0.0)
        info['cost_hazards'] = info.get('cost_hazards', 0.0)


        output = (state, reward, terminated, truncated, info)
        print(f"[STEP] Final output: {type(output)}, length={len(output)}")
        return output

    def render(self, **args):
        return self._env.render(**args)

    def close(self):
        self._env.close()


def Env(env_name, seed, max_episode_length=1000, action_repeat=1):
    print(f"[FACTORY] Creating environment wrapper for: {env_name}")
    env_list = ['Jackal-v0', 'Doggo-v0', 'Safexp-PointGoal1-v0', 'Safexp-CarGoal1-v0','SafetyPointGoal1-v0']
    if not env_name in env_list:
        raise ValueError(f'[FACTORY] Invalid environment name.\nSupport Env list: {env_list}') 
    else:
        return GymEnv(env_name, seed, max_episode_length, action_repeat)
