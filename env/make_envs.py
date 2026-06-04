import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
# subprocvecenv收集了reset过程中的infos但是并没有返回
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_ # TODO use this to vector envs like Mujoco and Cartpole


from typing import Any


import numpy as np
from omegaconf import DictConfig
from .env_types import ENVS_NAME_TYPE
from .atari_envs import atari_wrap
from .mujoco_envs import make_mujoco_env
from .basic_envs import make_basic_env
from utils.utils import RunningMeanStd


class VecObsNorm(VecEnvWrapper):
    """Normalize observations from a Stable-Baselines3 vectorized environment."""

    def __init__(self, envs, update_obs_rms=True):
        super().__init__(venv=envs)
        self.update_obs_rms = update_obs_rms
        self.obs_rms = RunningMeanStd()

    def reset(self):
        obs = self.venv.reset()
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs)
        return self._norm_obs(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs)
        obs = self._norm_obs(obs).astype(np.float32)

        # process with the infos 
        for idx, done in enumerate(dones):
            if done and "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self._norm_obs(
                    infos[idx]["terminal_observation"]
                ).astype(np.float32)

        return obs, rewards, dones, infos

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms:
            return self.obs_rms.norm(obs)
        return obs

    def set_obs_rms(self, obs_rms: RunningMeanStd) -> None:
        """Set with given observation running mean/std."""
        self.obs_rms = obs_rms

    def get_obs_rms(self) -> RunningMeanStd:
        """Return observation running mean/std."""
        return self.obs_rms




def make_env(args: DictConfig,is_training: bool,scale=False):
    env_name = args.name
    env_type = ENVS_NAME_TYPE[env_name]
    if env_type=="atari":
        env = gym.make(env_name, frameskip=1)
        env = atari_wrap(env, episode_life=is_training, clip_rewards=is_training, frame_stack=args.frame_stack, scale=scale,frame_skip=args.frame_skip)
    else:
        raise NotImplementedError(f"Environment {env_name} is not supported yet.")
    
    return env


def make_env_func(args: DictConfig, is_training: bool,scale=False,seed=None):
    def _thunk():
        # TODO: add more envs except for atari envs 
        env_name = args.name
        env_type = ENVS_NAME_TYPE[env_name]
        
        if env_type=="atari":
            env = gym.make(env_name, frameskip=1)
            env = atari_wrap(env, episode_life=is_training, clip_rewards=is_training, frame_stack=args.frame_stack, scale=scale,frame_skip=args.frame_skip)
        elif env_type=="mujoco":
            env = make_mujoco_env(env_name,max_episode_length=args.max_episode_length)
        elif env_type=="basic":
            env = make_basic_env(env_name,max_episode_length=args.max_episode_length)
        else:
            raise NotImplementedError(f"Environment {env_name} is not supported yet.")
        env.reset(seed=seed)
        env.action_space.seed(seed=seed)
        return env
    return _thunk

def make_vec_envs(args: DictConfig,is_training: bool,seed: int, scale=False):
    
    env_num = args.num_training_envs if is_training else args.num_testing_envs

    envs_func = [
        make_env_func(args,is_training,scale=scale,seed=i+seed)
        for i in range(env_num)
    ]
    # create vectorized environments, if env_num == 1, use DummyVecEnv to avoid unnecessary subprocesses
    # TODO:here the sb3 returns done = done or truncated, we may need to handle the truncated in the envs like absorbing states
    if env_num == 1:
        envs = DummyVecEnv(envs_func)
    else:
        envs = SubprocVecEnv(envs_func)

    # env_name = args.name
    # if ENVS_NAME_TYPE[env_name]=="mujoco" and args.obs_norm: 

    if getattr(args, "obs_norm", False):
        envs = VecObsNorm(envs, update_obs_rms=is_training)
    return envs





if __name__ == "__main__":
    # just use this to test the envs
    def make_easy_test():
        # TODO: add more envs except for atari envs 
        env = gym.make('ALE/Breakout-v5')
        env = atari_wrap(env, episode_life=True, clip_rewards=False, frame_stack=4, scale=False)
        return env
    
    env = gym.make('BreakoutNoFrameskip-v4')
    env = atari_wrap(env, episode_life=True, clip_rewards=False, frame_stack=4, scale=False)
    action = 3
    obs = env.reset()
    
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(last_info, info)
            input('terminate')
            env.reset()
        last_info = info


    exit()
    env_num = 2
    envs_func = [
        make_easy_test
        for i in range(env_num)
    ]

    envs = SubprocVecEnv(envs_func)
    # envs = DummyVecEnv(envs_func)

    obs = envs.reset()
    i = 0
    rewards = np.zeros(envs.num_envs)
    saved = []
    while True:
        actions = [2 for _ in range(envs.num_envs)]
        
        obs, reward, terminated, info = envs.step(actions)
        lives = [info[i]['lives'] for i in range(len(info))]
        
        rewards += rewards
        
        
        for j in range(envs.num_envs):
            if terminated[j]:
                print(terminated,[kk['lives'] for kk in info])
                input("continue")
                for kk in range(envs.num_envs):
                    envs.en[kk].unwrapped.reset()
                actions = [2 for _ in range(envs.num_envs)]
                obs, reward, terminated, info = envs.step(actions)
                print(terminated,info)
                exit()
            # if info[j]['lives']==0:
                
            #     input("over now")
            #     saved = np.array(saved)
            #     np.save("states",saved) 
            #     exit()
    # saved = np.array(saved)
    # np.save("states",saved)
    
