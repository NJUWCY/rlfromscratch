import numpy as np
import gymnasium as gym 
import ale_py
# register the atari envs to gymnasium
gym.register_envs(ale_py)

from gymnasium.wrappers import TimeLimit
from collections import deque
from typing import Any, SupportsFloat

from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, MaxAndSkipEnv,FireResetEnv, NoopResetEnv, WarpFrame
from .basic_envs import Monitor
from .basic_envs import TruncatedMonitor


def _parse_reset_result(reset_result: tuple) -> tuple[tuple, dict, bool]:
    contains_info = (
        isinstance(reset_result, tuple)
        and len(reset_result) == 2
        and isinstance(reset_result[1], dict)
    )
    if contains_info:
        return reset_result[0], reset_result[1], contains_info
    return reset_result, {}, contains_info


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to 0~1.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        low = np.min(obs_space.low)
        high = np.max(obs_space.high)
        # Note: if don't convert to float32, then the uint8 will be converted to float64, in later tensor and buffer cast will be slower 11x
        self.bias = np.float32(low)
        self.scale = np.float32(high - low)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation - self.bias) / self.scale

class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env: gym.Env, n_frames: int) -> None:
        super().__init__(env)
        self.n_frames: int = n_frames
        self.frames: deque[tuple[Any, ...]] = deque([], maxlen=n_frames)
        obs_space = env.observation_space
        obs_space_shape = env.observation_space.shape
        assert obs_space_shape is not None
        shape = (n_frames*obs_space_shape[0], *(obs_space_shape[1:]))
        assert isinstance(obs_space, gym.spaces.Box)
        obs_space_dtype = obs_space.dtype
        self.observation_space = gym.spaces.Box(
            low=np.min(obs_space.low),
            high=np.max(obs_space.high),
            shape=shape,
            dtype=obs_space_dtype,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return (self._get_ob(), info) if return_info else (self._get_ob(), {})

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_result = self.env.step(action)
        done: bool
        if len(step_result) == 4:
            obs, reward, done, info = step_result  # type: ignore[unreachable] # mypy doesn't know that Gym version <0.26 has only 4 items (no truncation)
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            new_step_api = True
        self.frames.append(obs)
        reward = float(reward)
        if new_step_api:
            return self._get_ob(), reward, term, trunc, info
        return (
            self._get_ob(),
            reward,
            done,
            info.get("TimeLimit.truncated", False),
            info,
        )

    def _get_ob(self) -> np.ndarray:
        # frames is a list of four (channels,84,84) arrays, we need to convert to (self.n_frames*channels,84,84)
        # TODO: consider other condition
        return np.concatenate(self.frames, axis=0)


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])




def atari_wrap(env, episode_life=True, clip_rewards=True, frame_stack=4, scale=False,frame_skip=4,max_episode_steps=10000):
    """Configure environment for DeepMind-style Atari.
    """
    env = NoopResetEnv(env, noop_max=30) # add random no-op action at the beginning of each episode to introduce randomness
    env = MaxAndSkipEnv(env, skip=frame_skip) # skip 4 frames and take the max of the last 2 frames to reduce computational cost and deal with flickering
    env = TimeLimit(env, max_episode_steps=max_episode_steps) # set a time limit to prevent infinite episodes, since some games can last forever
    
    env = Monitor(env)
    
    if episode_life:
        env = EpisodicLifeEnv(env) # make end of life == end of episode, but only reset on true game over, so that the agent learns to avoid losing lives
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    
    # transpose the image to channel-first format for PyTorch
    shape = env.observation_space.shape
    if len(shape) == 3 and shape[2] in [1, 3]:
        env = TransposeImage(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    
    env = TruncatedMonitor(env)

    return env

if __name__=="__main__":
    env = gym.make("BreakoutNoFrameskip-v4")
    env = atari_wrap(env)
    env.reset()
    done = False 
    i = 0
    while True :
        action = 0
        ns, r, done,t, info = env.step(action)
        i+=1 
        if 'episode' in info:
            break
        if done:
            env.reset()
    print(info)