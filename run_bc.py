"""Train a probabilistic policy on the library's fixed expert datasets."""

import logging
import os
from datetime import datetime

import hydra
import numpy as np
from gymnasium.spaces import Box, Discrete
from omegaconf import DictConfig, OmegaConf

from agent import PolicyAgent
from algorithm import ALGORITHM_DICT
from env.make_envs import make_env
from logger.logger import Logger
from memory import ExpertDataset
from utils import ACTIVATION_DICT
from utils.networks import (AtariCNNEncoder, DiagGaussianActor, DiscreteProbabilityActor,
                            MLPNetwork, TanhGaussianActor)
from utils.utils import get_best_device, set_seed


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_actor(observation_space, action_space, bc_args, device):
    actor_type = bc_args.actor_type
    if actor_type == "auto":
        actor_type = ("DiscreteProbabilityActor" if isinstance(action_space, Discrete)
                      else "TanhGaussianActor")
    activation = ACTIVATION_DICT[bc_args.activation]
    if actor_type == "DiscreteProbabilityActor":
        if not isinstance(action_space, Discrete) or action_space.start != 0:
            raise ValueError("DiscreteProbabilityActor requires zero-based Discrete actions")
        if bc_args.common_net:
            raise ValueError("common_net only applies to TanhGaussianActor")
        if len(observation_space.shape) == 1:
            encoder = MLPNetwork(observation_space.shape[0], 0,
                                 hidden_sizes=bc_args.hidden_sizes,
                                 activation=activation, initialize=bc_args.initialize)
            encoder.embedding_dim = (bc_args.hidden_sizes[-1] if bc_args.hidden_sizes
                                     else observation_space.shape[0])
        elif len(observation_space.shape) == 3:
            encoder = AtariCNNEncoder(observation_space.shape, bc_args.embedding_dim,
                                      initialize=bc_args.initialize)
        else:
            raise ValueError("BC supports vector or channel-first image observations for discrete actions")
        return DiscreteProbabilityActor(observation_space, action_space, encoder,
                                        initialize=bc_args.initialize)
    if actor_type not in ("DiagGaussianActor", "TanhGaussianActor"):
        raise ValueError("BC actor_type must be auto, DiagGaussianActor, "
                         "TanhGaussianActor or DiscreteProbabilityActor")
    if (not isinstance(action_space, Box) or len(action_space.shape) != 1
            or len(observation_space.shape) != 1):
        raise ValueError("Gaussian BC actors require vector Box observations and actions")
    options = dict(
        device=device, hidden_sizes=bc_args.hidden_sizes, activation=activation,
        initialize=bc_args.initialize, state_dependent_std=bc_args.state_dependent_std,
        initial_log_sigma=bc_args.initial_log_sigma,
    )
    if actor_type == "TanhGaussianActor":
        if (not np.isfinite(action_space.low).all() or not np.isfinite(action_space.high).all()
                or np.any(action_space.high <= action_space.low)):
            raise ValueError("TanhGaussianActor requires finite nonzero action bounds")
        return TanhGaussianActor(observation_space, action_space, MLPNetwork,
                                 common_net=bc_args.common_net, **options)
    if bc_args.common_net:
        raise ValueError("DiagGaussianActor requires common_net=false")
    return DiagGaussianActor(observation_space, action_space, MLPNetwork, **options)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(args: DictConfig):
    if args.algorithm.name != "BC":
        raise ValueError("Run behavior cloning with algorithm=bc")
    if args.env.absorbing:
        raise ValueError("BC requires env.absorbing=false")
    args.hydra_base_dir = os.getcwd()
    set_seed(args.seed)
    device = get_best_device()

    # Construct only to inspect spaces; do not reset or step this environment.
    env = make_env(args.env, is_training=False, scale=args.env.scale)
    try:
        observation_space, action_space = env.observation_space, env.action_space
    finally:
        env.close()

    actor = build_actor(observation_space, action_space, args.algorithm, device)
    agent = PolicyAgent(observation_space, action_space, device, actor)
    dataset = ExpertDataset(
        args.expert_dataset.data_path, observation_space, action_space,
        args.expert_dataset.trajectory_num, args.expert_dataset.subsample_frequency,
        gamma=args.gamma, nstep=1, absorbing=False,
        random_start=args.expert_dataset.random_start,
    )
    os.makedirs(args.log_dir, exist_ok=True)
    logger = Logger(
        project_name=args.experiment_name,
        run_name=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config=args, log_dir=args.log_dir, use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard, use_swanlab=args.use_swanlab,
        step_axis="gradient",
    )
    try:
        algorithm = ALGORITHM_DICT["BC"](
            training_envs=None, testing_envs=None, buffer=dataset,
            agent=agent, logger=logger, device=device, args=args,
            bc_args=args.algorithm,
        )
    except Exception:
        logger.close()
        raise
    logging.info("Begin offline BC: %s, %d fixed transitions",
                 actor.__class__.__name__, algorithm.dataset_size)
    algorithm.run()


if __name__ == "__main__":
    main()
