import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


def absorbing_state(observation_dim: int, dtype=np.float32) -> np.ndarray:
    """The DAC absorbing state: all features zeroed, indicator bit set."""
    state = np.zeros(observation_dim, dtype=dtype)
    state[-1] = 1.0
    return state


def absorbing_action(action_dim: int, dtype=np.float32) -> np.ndarray:
    """The action the agent is forced to take at the absorbing state."""
    return np.zeros(action_dim, dtype=dtype)


def insert_absorbing_self_loops(
    states: np.ndarray,
    next_states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    truncateds: np.ndarray,
    absorbing_obs: np.ndarray,
    absorbing_act: np.ndarray,
    extras: dict | None = None,
) -> dict:
    """Redirect every real termination to the absorbing state and insert a self-loop after it.

    ``extras`` maps a field name to ``(array, insert_value)`` so callers can
    carry extra per-step arrays (e.g. log-probs) through the same insertion.
    """
    states = np.asarray(states, dtype=np.float32).copy()
    next_states = np.asarray(next_states, dtype=np.float32).copy()
    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1).copy()
    dones = np.asarray(dones, dtype=np.float32).reshape(-1).copy()
    truncateds = np.asarray(truncateds, dtype=np.float32).reshape(-1).copy()
    extra_values = {key: np.asarray(value[0]).copy() for key, value in (extras or {}).items()}
    extra_inserts = {key: value[1] for key, value in (extras or {}).items()}

    length = len(rewards)
    is_absorbing = np.zeros(length, dtype=np.float32)
    next_is_absorbing = np.zeros(length, dtype=np.float32)

    terminal = np.logical_and(dones > 0, truncateds == 0)
    if terminal.any():
        next_states[terminal] = absorbing_obs
        dones[terminal] = 0.0
        next_is_absorbing[terminal] = 1.0

        extra = terminal.astype(np.int64)
        old_to_new = np.arange(length) + np.cumsum(extra) - extra
        insert_at = old_to_new[terminal] + 1
        new_length = length + int(extra.sum())

        def _place(old, insert_value):
            old = np.asarray(old)
            out = np.zeros((new_length,) + old.shape[1:], dtype=old.dtype)
            out[old_to_new] = old
            out[insert_at] = insert_value
            return out

        states = _place(states, absorbing_obs)
        next_states = _place(next_states, absorbing_obs)
        actions = _place(actions, absorbing_act)
        rewards = _place(rewards, np.float32(0.0))
        dones = _place(dones, np.float32(0.0))
        truncateds = _place(truncateds, np.float32(0.0))
        is_absorbing = _place(is_absorbing, np.float32(1.0))
        next_is_absorbing = _place(next_is_absorbing, np.float32(1.0))
        extra_values = {key: _place(extra_values[key], extra_inserts[key]) for key in extra_values}

    result = {
        "states": states,
        "next_states": next_states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "truncateds": truncateds,
        "absorbing": is_absorbing,
        "next_absorbing": next_is_absorbing,
    }
    result.update(extra_values)
    return result


class AbsorbingWrapper(gym.Wrapper):
    """Append the DAC absorbing-state indicator bit to every observation.

    The indicator is always 0 on observations produced by the environment. The
    absorbing state itself is synthesised by the replay buffer / expert dataset
    when an episode ends in a real termination (not a time limit), so that the
    discriminator sees the same state distribution for expert and policy data
    and the critic can learn a value for "being dead".
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        observation_space = env.observation_space
        assert len(observation_space.shape) == 1, "AbsorbingWrapper only supports 1D observations"
        self.observation_space = Box(
            low=np.concatenate([observation_space.low, np.zeros(1, dtype=observation_space.dtype)]),
            high=np.concatenate([observation_space.high, np.ones(1, dtype=observation_space.dtype)]),
            dtype=observation_space.dtype,
        )

    def _extend(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate([observation, np.zeros(1, dtype=observation.dtype)])

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._extend(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "truncated_observation" in info:
            info["truncated_observation"] = self._extend(info["truncated_observation"])
        return self._extend(observation), reward, terminated, truncated, info
