from pathlib import Path

import gymnasium as gym
import torch
from omegaconf import DictConfig

from env.env_types import ENVS_NAME_TYPE
from utils import ACTIVATION_DICT
from utils.networks import (
    AtariCNNEncoder,
    AtariDQNNetwork,
    DeterministicActor,
    DiagGaussianActor,
    DiscreteProbabilityActor,
    DiscreteVFunction,
    DoubleQFunction,
    MLPNetwork,
    QFunction,
    TanhDeterministicActor,
    TanhGaussianActor,
    ValueFunction,
)

from .agent import AgentBase, AtariDQNAgent, ProbabilityA2CAgent, SACAgent, TD3Agent


def _config_value(config: DictConfig, name: str, default):
    return getattr(config, name, default)


def _build_continuous_actor(
    algorithm_config: DictConfig,
    observation_space: gym.spaces.Box,
    action_space: gym.spaces.Box,
    device: torch.device,
):
    kwargs = {
        "device": device,
        "state_dependent_std": _config_value(
            algorithm_config, "state_dependent_std", False
        ),
        "hidden_sizes": list(_config_value(algorithm_config, "hidden_sizes", [64, 64])),
        "activation": torch.nn.Tanh,
        "initialize": _config_value(algorithm_config, "initialize", False),
    }
    if not _config_value(algorithm_config, "rescale", False):
        return DiagGaussianActor(
            observation_space,
            action_space,
            MLPNetwork,
            **kwargs,
        )
    if _config_value(algorithm_config, "action_bound_method", "tanh") != "tanh":
        raise ValueError("Only tanh action rescaling is supported for expert rollout.")
    return TanhGaussianActor(
        observation_space,
        action_space,
        MLPNetwork,
        **kwargs,
    )


def _build_value_function(
    algorithm_config: DictConfig,
    observation_space: gym.spaces.Box,
) -> ValueFunction:
    return ValueFunction(
        observation_space,
        MLPNetwork,
        hidden_sizes=list(_config_value(algorithm_config, "hidden_sizes", [64, 64])),
        activation=torch.nn.Tanh,
        initialize=_config_value(algorithm_config, "initialize", False),
    )


def _build_q_function(
    algorithm_config: DictConfig,
    observation_space: gym.spaces.Box,
    action_space: gym.spaces.Box,
):
    q_function_type = (
        DoubleQFunction
        if _config_value(algorithm_config, "double_critic", True)
        else QFunction
    )
    activation_name = _config_value(algorithm_config, "activation", "relu")
    try:
        activation = ACTIVATION_DICT[activation_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {activation_name!r}.") from exc
    return q_function_type(
        observation_space,
        action_space,
        MLPNetwork,
        hidden_sizes=list(_config_value(algorithm_config, "hidden_sizes", [256, 256])),
        activation=activation,
    )


def build_expert_agent(
    cfg: DictConfig,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    device: torch.device,
) -> AgentBase:
    if _config_value(cfg.env, "obs_norm", False):
        raise ValueError(
            "Expert rollout does not support obs_norm=true because checkpoints do not "
            "contain the training observation statistics."
        )

    algorithm_name = str(cfg.algorithm.name).upper()
    env_name = str(cfg.env.name)
    env_type = ENVS_NAME_TYPE.get(env_name)
    supported_pairs = {
        "DQN": {"atari"},
        "PPO": {"atari", "mujoco"},
        "TRPO": {"mujoco"},
        "SAC": {"mujoco"},
        "TD3": {"mujoco"},
    }
    if algorithm_name not in supported_pairs:
        raise ValueError(f"Unsupported expert algorithm {algorithm_name!r}.")
    if env_type not in supported_pairs[algorithm_name]:
        readable_env_type = "Atari" if env_type == "atari" else env_type or env_name
        raise ValueError(
            f"{algorithm_name} expert rollout is not supported for {readable_env_type}."
        )

    algorithm_config = cfg.algorithm
    if algorithm_name == "DQN":
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("DQN expert rollout requires a discrete action space.")
        network = AtariDQNNetwork(
            observation_space.shape,
            action_space.n,
            dueling_network=_config_value(
                algorithm_config, "dueling_network", False
            ),
            conv_gradient_rescale=_config_value(
                algorithm_config, "conv_gradient_rescale", False
            ),
        )
        agent = AtariDQNAgent(
            observation_space,
            action_space,
            device,
            network=network,
        )
    elif algorithm_name == "PPO" and env_type == "atari":
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("Atari PPO expert rollout requires a discrete action space.")
        encoder = AtariCNNEncoder(
            observation_space.shape,
            512,
            initialize=_config_value(algorithm_config, "initialize", False),
        )
        actor = DiscreteProbabilityActor(
            observation_space,
            action_space,
            encoder,
            initialize=_config_value(algorithm_config, "initialize", False),
        )
        critic = DiscreteVFunction(
            observation_space,
            action_space,
            encoder,
            initialize=_config_value(algorithm_config, "initialize", False),
        )
        agent = ProbabilityA2CAgent(
            observation_space,
            action_space,
            device,
            actor,
            critic,
        )
    else:
        if not isinstance(observation_space, gym.spaces.Box) or not isinstance(
            action_space, gym.spaces.Box
        ):
            raise ValueError(
                f"{algorithm_name} MuJoCo expert rollout requires Box spaces."
            )

        if algorithm_name in {"PPO", "TRPO"}:
            actor = _build_continuous_actor(
                algorithm_config,
                observation_space,
                action_space,
                device,
            )
            agent = ProbabilityA2CAgent(
                observation_space,
                action_space,
                device,
                actor,
                _build_value_function(algorithm_config, observation_space),
            )
        elif algorithm_name == "SAC":
            actor = _build_continuous_actor(
                algorithm_config,
                observation_space,
                action_space,
                device,
            )
            critic = _build_q_function(
                algorithm_config,
                observation_space,
                action_space,
            )
            use_target = _config_value(algorithm_config, "use_target", False)
            target_critic = (
                _build_q_function(
                    algorithm_config,
                    observation_space,
                    action_space,
                )
                if use_target
                else None
            )
            agent = SACAgent(
                observation_space,
                action_space,
                device,
                actor,
                critic,
                target_critic=target_critic,
                use_target=use_target,
                target_update_method=_config_value(
                    algorithm_config, "target_update_method", "soft"
                ),
                target_update_tau=_config_value(
                    algorithm_config, "target_update_tau", 0.005
                ),
                double_critic=_config_value(
                    algorithm_config, "double_critic", True
                ),
            )
        else:
            activation_name = _config_value(algorithm_config, "activation", "relu")
            try:
                activation = ACTIVATION_DICT[activation_name]
            except KeyError as exc:
                raise ValueError(f"Unsupported activation {activation_name!r}.") from exc
            actor_type = (
                TanhDeterministicActor
                if _config_value(algorithm_config, "rescale", False)
                else DeterministicActor
            )
            actor_kwargs = {
                "observation_space": observation_space,
                "action_space": action_space,
                "net_architecture": MLPNetwork,
                "device": device,
                "explore_noise_sigma": _config_value(
                    algorithm_config, "explore_noise_sigma", 0.1
                ),
                "hidden_sizes": list(
                    _config_value(algorithm_config, "hidden_sizes", [256, 256])
                ),
                "activation": activation,
            }
            actor = actor_type(**actor_kwargs)
            target_actor = actor_type(**actor_kwargs)
            critic = _build_q_function(
                algorithm_config,
                observation_space,
                action_space,
            )
            use_target = _config_value(algorithm_config, "use_target", False)
            target_critic = (
                _build_q_function(
                    algorithm_config,
                    observation_space,
                    action_space,
                )
                if use_target
                else None
            )
            agent = TD3Agent(
                observation_space,
                action_space,
                device,
                actor,
                critic,
                target_actor=target_actor,
                target_critic=target_critic,
                use_target=use_target,
                target_update_method=_config_value(
                    algorithm_config, "target_update_method", "soft"
                ),
                target_update_tau=_config_value(
                    algorithm_config, "target_update_tau", 0.005
                ),
                double_critic=_config_value(
                    algorithm_config, "double_critic", True
                ),
            )

    return agent.to(device)


def load_policy_weights(
    agent: AgentBase,
    checkpoint_path: str | Path,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # build_expert_agent reconstructs the full agent (policy + critic/targets) from
    # the same saved config, so agent.load can restore the whole state_dict directly
    # and also re-apply the saved observation/action spaces.
    try:
        agent.load(str(checkpoint_path))
    except (KeyError, RuntimeError) as exc:
        raise RuntimeError(
            f"Failed to load policy weights from {checkpoint_path}: {exc}"
        ) from exc
