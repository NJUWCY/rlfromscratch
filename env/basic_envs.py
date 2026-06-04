import numpy as np
import gymnasium as gym 
from gymnasium.wrappers import TimeLimit
from typing import Any


class Monitor(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(
        self,
        env: gym.Env
    ):
        super().__init__(env=env)
        self.rewards: list[float] = []
        

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        self.rewards = []

        return self.env.reset(**kwargs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        if terminated or truncated:
            ep_rew = sum(self.rewards)
            ep_info = {"r": round(ep_rew, 6)}
            info["episode"] = ep_info
            if not terminated and truncated:
                info['truncated'] = True 
                info['last_observation'] = observation
        return observation, reward, terminated, truncated, info

def make_basic_env(env_name:str, max_episode_length=1000):
    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=max_episode_length)
    env = Monitor(env)
    
    return env