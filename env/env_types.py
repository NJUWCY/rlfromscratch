import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
atari_envs = sorted([
    env_id
    for env_id in gym.envs.registry.keys()
    if env_id.endswith("NoFrameskip-v4")
])

mujoco_envs = sorted([
    env_id
    for env_id in gym.envs.registry.keys()
    if any(name in env_id for name in [
        "HalfCheetah",
        "Hopper",
        "Walker2d",
        "Ant",
        "Humanoid",
        "Swimmer",
        "Reacher",
        "Pusher",
        "InvertedPendulum",
        "InvertedDoublePendulum",
    ])
])

basic_envs = sorted([
    env_id
    for env_id in gym.envs.registry.keys()
    if any(name in env_id for name in [
        "Pendulum", "MountainCarContinuous"
    ])
])

ENVS_TYPE_NAME = {
    "atari": atari_envs,
    "mujoco" : mujoco_envs,
    "basic": basic_envs
}

ENVS_NAME_TYPE = {
    v: k
    for k, lst in ENVS_TYPE_NAME.items()
    for v in lst
}