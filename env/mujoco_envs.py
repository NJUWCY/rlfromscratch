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
        return observation, reward, terminated, truncated, info




def make_mujoco_env(env_name:str, max_episode_length=1000):
    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=max_episode_length)
    env = Monitor(env)
    
    return env

if __name__ == "__main__":
    from stable_baselines3.common.vec_env import DummyVecEnv
    def func(l=100):
        return make_mujoco_env("Hopper-v5",l)
    
    env = DummyVecEnv([func])


    # env = gym.make("Hopper-v5",max_episode_steps=100)
    # env = Monitor(env)
    env.reset()
    re = 0
    for i in range(1001):
        ns, r, d, t, info = env.step(np.array([0,0,0]))
        print(d,r)
        re+=r
        if d:
            print(i,re)