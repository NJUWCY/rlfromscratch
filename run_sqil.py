"""Train SQIL with the project's environments, expert datasets and SAC agent."""

import copy
import logging
import os
from datetime import datetime

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from agent import SACAgent
from algorithm import ALGORITHM_DICT
from env.make_envs import make_vec_envs
from logger.logger import Logger
from memory import ExpertDataset, ReplayBuffer
from utils import ACTIVATION_DICT
from utils.networks import DoubleQFunction, MLPNetwork, QFunction, TanhGaussianActor
from utils.utils import get_best_device, set_seed


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    args = cfg
    args.hydra_base_dir = os.getcwd()
    print("Parameters:", OmegaConf.to_yaml(args))
    if args.algorithm.name != "SQIL":
        raise ValueError("Run SQIL with algorithm=sqil")
    rl_args = args.algorithm.rl
    if int(rl_args.nstep) != 1:
        raise ValueError("SQIL currently supports nstep=1")
    if int(rl_args.batch_size) < 1:
        raise ValueError("SQIL requires batch_size >= 1")
    if args.env.absorbing:
        raise ValueError("SQIL requires env.absorbing=false")
    if rl_args.action_bound_method != "tanh" or not rl_args.rescale:
        raise ValueError("SQIL uses action_bound_method=tanh and rescale=true (environment action units)")
    if rl_args.buffer_name != "ReplayBuffer":
        raise ValueError("SQIL uses ReplayBuffer for online transitions")
    capacity_per_env = int(rl_args.buffer_size) // int(args.env.num_training_envs)
    if capacity_per_env < int(args.interact_per_epoch):
        raise ValueError("SQIL buffer capacity per environment must be >= interact_per_epoch")
    if args.interact_per_epoch <= 0 or args.update_step_per_epoch <= 0:
        raise ValueError("SQIL collection and update counts must be positive")

    set_seed(args.seed)
    device = get_best_device()
    training_envs = make_vec_envs(args.env, True, seed=args.seed)
    try:
        buffer = ReplayBuffer(
            training_envs.observation_space, training_envs.action_space,
            capacity_per_env, num_envs=args.env.num_training_envs,
            gamma=args.gamma, nstep=rl_args.nstep,
        )
        activation = ACTIVATION_DICT[rl_args.activation]
        actor = TanhGaussianActor(
            training_envs.observation_space, training_envs.action_space,
            MLPNetwork, device=device,
            state_dependent_std=rl_args.state_dependent_std,
            hidden_sizes=rl_args.hidden_sizes, activation=activation,
            initialize=rl_args.initialize, common_net=rl_args.common_net,
        )
        critic_class = DoubleQFunction if rl_args.double_critic else QFunction
        critic = critic_class(
            training_envs.observation_space, training_envs.action_space,
            MLPNetwork, hidden_sizes=rl_args.hidden_sizes,
            activation=activation, initialize=rl_args.initialize,
        )
        target_critic = copy.deepcopy(critic) if rl_args.use_target else None
        agent = SACAgent(
            training_envs.observation_space, training_envs.action_space,
            device, actor, critic, target_critic=target_critic,
            use_target=rl_args.use_target,
            target_update_method=rl_args.target_update_method,
            target_update_tau=rl_args.target_update_tau,
            double_critic=rl_args.double_critic,
        )

        logging.info("Loading the existing expert dataset...")
        expert_buffer = ExpertDataset(
            args.expert_dataset.data_path,
            training_envs.observation_space, training_envs.action_space,
            args.expert_dataset.trajectory_num,
            args.expert_dataset.subsample_frequency,
            gamma=args.gamma, nstep=1, absorbing=False,
            random_start=args.expert_dataset.random_start,
        )
        expert_actions = expert_buffer.buffer["actions"]
        if (not np.isfinite(expert_actions).all()
                or np.any(expert_actions < training_envs.action_space.low - 1e-5)
                or np.any(expert_actions > training_envs.action_space.high + 1e-5)):
            raise ValueError("Expert actions must be in environment coordinates and within its action bounds")

        os.makedirs(args.log_dir, exist_ok=True)
        logger = Logger(
            project_name=args.experiment_name,
            run_name=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config=args, log_dir=args.log_dir,
            use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard,
            use_swanlab=args.use_swanlab,
        )
        algorithm = ALGORITHM_DICT[args.algorithm.name](
            training_envs=training_envs, testing_envs=None, buffer=buffer,
            expert_buffer=expert_buffer, agent=agent, logger=logger,
            device=device, args=args, rl_args=rl_args,
        )
        logging.info("Begin SQIL training: expert reward=1, online reward=0, sampling_schedule=%s",
                     algorithm.sampling_schedule.tolist())
        algorithm.run()
    finally:
        training_envs.close()


if __name__ == "__main__":
    main()
