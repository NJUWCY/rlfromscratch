import numpy as np
import gymnasium as gym 
from gymnasium.wrappers import TimeLimit
from typing import Any
from .basic_envs import Monitor
from .basic_envs import TruncatedMonitor


class MuJoCoStateInfo(gym.Wrapper):
    """Inject simulator ``qpos`` / ``qvel`` into ``info``.

    Values correspond to the **pre-step** state so they align with the
    ``observations`` entry of the same transition (D4RL convention).
    """

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        info["qpos"] = np.array(self.unwrapped.data.qpos, dtype=np.float64, copy=True)
        info["qvel"] = np.array(self.unwrapped.data.qvel, dtype=np.float64, copy=True)
        return observation, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        qpos = np.array(self.unwrapped.data.qpos, dtype=np.float64, copy=True)
        qvel = np.array(self.unwrapped.data.qvel, dtype=np.float64, copy=True)
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["qpos"] = qpos
        info["qvel"] = qvel
        return observation, reward, terminated, truncated, info


def make_mujoco_env(env_name:str, max_episode_length=1000):
    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=max_episode_length)
    env = MuJoCoStateInfo(env)
    env = Monitor(env)
    env = TruncatedMonitor(env)
    
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
