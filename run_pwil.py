"""Train PWIL-SAC with the library's MuJoCo environments and expert datasets."""

import copy
import logging
import os
from datetime import datetime

import hydra
import numpy as np
from gymnasium.spaces import Box
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
def main(args: DictConfig):
    args.hydra_base_dir = os.getcwd()
    if args.algorithm.name != "PWIL":
        raise ValueError("Run PWIL with algorithm=pwil")
    rl_args = args.algorithm.rl
    if args.env.absorbing:
        raise ValueError("PWIL requires env.absorbing=false")
    if int(rl_args.nstep) != 1:
        raise ValueError("PWIL-SAC currently supports nstep=1")
    if rl_args.action_bound_method != "tanh" or not rl_args.rescale:
        raise ValueError("PWIL-SAC uses tanh/rescale=true and environment action coordinates")
    if rl_args.buffer_name != "ReplayBuffer":
        raise ValueError("PWIL uses the existing ReplayBuffer")
    if min(args.env.num_training_envs, args.interact_per_epoch,
           args.update_step_per_epoch, rl_args.batch_size) < 1:
        raise ValueError("PWIL environment, collection, update and batch counts must be positive")
    capacity = int(rl_args.buffer_size) // int(args.env.num_training_envs)
    if capacity < int(args.interact_per_epoch):
        raise ValueError("Replay capacity per environment must be >= interact_per_epoch")

    set_seed(args.seed)
    device = get_best_device()
    # PWIL must see raw observations before policy normalization. The algorithm
    # adds the reward wrapper, then an optional frozen observation normalizer.
    env_args = OmegaConf.create(OmegaConf.to_container(args.env, resolve=True))
    assert env_args.obs_norm==False, "PWIL requires obs_norm=false, otherwise the rewarder sees normalized observations"
    training_envs = make_vec_envs(env_args, True, seed=args.seed)
    logger = None
    try:
        observation_space, action_space = training_envs.observation_space, training_envs.action_space
        if (not isinstance(observation_space, Box) or len(observation_space.shape) != 1
                or not isinstance(action_space, Box) or len(action_space.shape) != 1
                or not np.isfinite(action_space.low).all()
                or not np.isfinite(action_space.high).all()
                or np.any(action_space.high <= action_space.low)):
            raise ValueError("PWIL-SAC requires vector observations and finite vector Box actions")
        buffer = ReplayBuffer(observation_space, action_space, capacity,
                              args.env.num_training_envs, gamma=args.gamma, nstep=1)
        activation = ACTIVATION_DICT[rl_args.activation]
        actor = TanhGaussianActor(
            observation_space, action_space, MLPNetwork, device=device,
            hidden_sizes=rl_args.hidden_sizes, activation=activation,
            state_dependent_std=rl_args.state_dependent_std,
            initialize=rl_args.initialize, common_net=rl_args.common_net,
        )
        critic_class = DoubleQFunction if rl_args.double_critic else QFunction
        critic = critic_class(observation_space, action_space, MLPNetwork,
                              hidden_sizes=rl_args.hidden_sizes, activation=activation,
                              initialize=rl_args.initialize)
        agent = SACAgent(
            observation_space, action_space, device, actor, critic,
            target_critic=copy.deepcopy(critic) if rl_args.use_target else None,
            use_target=rl_args.use_target, target_update_method=rl_args.target_update_method,
            target_update_tau=rl_args.target_update_tau, double_critic=rl_args.double_critic,
        )
        # ExpertDataset performs the only subsampling step. PWIL uses
        # this entire expert buffer for both rewards and replay prefill.
        expert_buffer = ExpertDataset(
            args.expert_dataset.data_path, observation_space, action_space,
            args.expert_dataset.trajectory_num,
            subsample_frequency=args.expert_dataset.subsample_frequency,
            gamma=args.gamma, nstep=1, absorbing=False,
            random_start=args.expert_dataset.random_start,
        )
        actions = expert_buffer.buffer["actions"]
        if (not np.isfinite(actions).all()
                or np.any(actions < action_space.low - 1e-5)
                or np.any(actions > action_space.high + 1e-5)):
            raise ValueError("PWIL expert actions must be within the environment action bounds")
        os.makedirs(args.log_dir, exist_ok=True)
        logger = Logger(
            project_name=args.experiment_name,
            run_name=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config=args, log_dir=args.log_dir, use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard, use_swanlab=args.use_swanlab,
        )
        algorithm = ALGORITHM_DICT["PWIL"](
            training_envs=training_envs, testing_envs=None, buffer=buffer,
            expert_buffer=expert_buffer, agent=agent, logger=logger,
            device=device, args=args, rl_args=rl_args,
        )
        logging.info("Begin PWIL-SAC: %d expert atoms, horizon=%d, alpha=%g, beta=%g, prefill=%d",
                     algorithm.rewarder.num_atoms, algorithm.rewarder.time_horizon,
                     algorithm.rewarder.alpha, algorithm.rewarder.beta,
                     algorithm.num_transitions_rb)
        algorithm.run()
        # BaseAlgorithm periodically saves; also preserve the final policy.
        algorithm.save(pre_fix="newest")
    finally:
        training_envs.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
