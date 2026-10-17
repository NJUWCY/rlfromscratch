import numpy as np
import gymnasium as gym 
from gymnasium.wrappers import TimeLimit
from typing import Any


class Monitor(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(
        self,
        env: gym.Env,
        collect_init_states: bool = False,
    ):
        super().__init__(env=env)
        self.rewards: list[float] = []
        self.collect_init_states = collect_init_states
        self.initial_states: list[np.ndarray] = []
        

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        self.rewards = []

        observation, info = self.env.reset(**kwargs)
        if self.collect_init_states:
            self.initial_states.append(np.asarray(observation, dtype=np.float32).copy())
        return observation, info

    def pop_initial_states(self):
        """Return raw reset observations since the last read, then clear them."""
        states = self.initial_states
        self.initial_states = []
        return states

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        if terminated or truncated:
            ep_rew = sum(self.rewards)
            ep_info = {"r": round(ep_rew, 6)}
            info["episode"] = ep_info
        return observation, reward, terminated, truncated, info
    
class TruncatedMonitor(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(
        self,
        env: gym.Env
    ):
        super().__init__(env=env)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        if (not terminated) and truncated:
            info['truncated'] = True 
            info['truncated_observation'] = observation
        return observation, reward, terminated, truncated, info

def make_basic_env(env_name:str, max_episode_length=1000, collect_init_states=False):
    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=max_episode_length)
    env = Monitor(env, collect_init_states=collect_init_states)
    
    return env