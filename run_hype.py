import os 
import logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] ---------------%(message)s---------------',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import hydra 
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from datetime import datetime 

from env.make_envs import make_vec_envs
from algorithm import ALGORITHM_DICT, HyPE, HyPESAC
from utils.utils import get_best_device, set_seed
from logger.logger import Logger 
from memory import AbsorbingReplayBuffer, ExpertDataset
from agent import SACAgent
from utils.networks import MLPNetwork, DoubleQFunction, QFunction, TanhGaussianActor
from discriminator.discriminator import GAILDiscriminator
from utils import ACTIVATION_DICT


def get_args(cfg: DictConfig):
    
    cfg.hydra_base_dir = os.getcwd()
    print("Parameters:", OmegaConf.to_yaml(cfg))
    return cfg


def build_discriminator_network(input_dim, discriminator_args):
    """Initialize the cost network using discriminator parameters only.

    default keeps Linear defaults. orthogonal_relu uses the library's
    sqrt(2) hidden gain and zero biases. orthogonal uses gain-1 hidden weights
    and default biases. Both orthogonal modes use discriminator_output_gain
    for the output weights.
    """
    mode = getattr(discriminator_args, "discriminator_initialization", "orthogonal_relu")
    if mode not in {"default", "orthogonal_relu", "orthogonal"}:
        raise ValueError("discriminator_initialization must be default, orthogonal_relu or orthogonal")
    spectral_norm = bool(getattr(discriminator_args, "spectral_norm", False))
    network = MLPNetwork(
        input_dim, 1,
        hidden_sizes=discriminator_args.discriminator_hidden_sizes,
        activation=ACTIVATION_DICT[discriminator_args.discriminator_activation],
        initialize=mode == "orthogonal_relu",
        last_std=discriminator_args.discriminator_output_gain,
        spectral_norm=spectral_norm and mode != "orthogonal",
    )
    if mode == "orthogonal":
        # Initialize after constructing all layers, as in garage; preserve biases.
        layers = [layer for layer in network.modules() if isinstance(layer, torch.nn.Linear)]
        for index, layer in enumerate(layers):
            gain = discriminator_args.discriminator_output_gain if index == len(layers) - 1 else 1.0
            torch.nn.init.orthogonal_(layer.weight, gain=gain)
        # Wrap only after initialization so spectral normalization sees the new weights.
        if spectral_norm:
            for layer in layers:
                torch.nn.utils.parametrizations.spectral_norm(layer)
    return network

@hydra.main(config_path="config", config_name="config",version_base="1.3")
def main(cfg: DictConfig):
    args = get_args(cfg)
    if args.algorithm.name not in {"HyPE", "HyPESAC"}:
        raise ValueError("Run HyPE with algorithm=hypesac")
    if int(args.algorithm.rl.nstep) != 1:
        raise ValueError("HyPE currently supports nstep=1")

    set_seed(args.seed)

    training_envs = make_vec_envs(args.env,True,seed=args.seed)

    logging.info("Checking for available GPUs...")
    device = get_best_device()

    logging.info("Creating the Logger...")
    runtime = datetime.now()
    runtime = runtime.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(args.log_dir, exist_ok=True)
    logger = Logger(project_name=args.experiment_name, run_name=runtime, config=args, log_dir=args.log_dir, use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard, use_swanlab=args.use_swanlab)

    rl_args = args.algorithm.rl
    discriminator_args = args.algorithm.discriminator

    logging.info("Creating the ReplayBuffer...")
    buffer = AbsorbingReplayBuffer(training_envs.observation_space,
                                   training_envs.action_space,
                                   rl_args.buffer_size,
                                   gamma=args.gamma,
                                   nstep=int(getattr(rl_args, "nstep", 1)),
                                   num_envs=args.env.num_training_envs)

    logging.info("Creating the Agent...")
    activation = ACTIVATION_DICT[rl_args.activation]
    # Keep the library actor/critic initialization, including output gain 0.01.
    orthogonal_init = bool(getattr(rl_args, "initialize", False))
    if rl_args.action_bound_method != "tanh":
        raise ValueError(f"HyPE requires a squashed Gaussian policy, got action_bound_method={rl_args.action_bound_method}")
    actor = TanhGaussianActor(
        training_envs.observation_space,
        training_envs.action_space,
        MLPNetwork,
        device=device,
        state_dependent_std=rl_args.state_dependent_std,
        hidden_sizes=rl_args.hidden_sizes,
        activation=activation,
        initialize=orthogonal_init,
        common_net=bool(getattr(rl_args, "common_net", False))
    )

    critic_class = DoubleQFunction if rl_args.double_critic else QFunction
    critic_kwargs = dict(hidden_sizes=rl_args.hidden_sizes, activation=activation, initialize=orthogonal_init)
    critic = critic_class(training_envs.observation_space, training_envs.action_space, MLPNetwork, **critic_kwargs)
    target_critic = critic_class(training_envs.observation_space, training_envs.action_space, MLPNetwork, **critic_kwargs) if rl_args.use_target else None

    agent = SACAgent(
        training_envs.observation_space,
        training_envs.action_space,
        device,
        actor,
        critic,
        target_critic=target_critic,
        use_target=rl_args.use_target,
        target_update_method=rl_args.target_update_method,
        target_update_tau=rl_args.target_update_tau,
        double_critic=rl_args.double_critic)

    logging.info("Creating the Discriminator...")
    discriminator_net = build_discriminator_network(
        training_envs.observation_space.shape[0] + training_envs.action_space.shape[0],
        discriminator_args,
    )

    discriminator = GAILDiscriminator(training_envs.observation_space,
                                      training_envs.action_space,
                                      device,
                                      discriminator_net,
                                      reward_function=discriminator_args.reward_function)

    logging.info("Loading the Expert Dataset...")
    expert_buffer = ExpertDataset(args.expert_dataset.data_path,
                                  training_envs.observation_space,
                                  training_envs.action_space,
                                  args.expert_dataset.trajectory_num,
                                  args.expert_dataset.subsample_frequency,
                                  args.gamma,
                                  1,
                                  absorbing=args.env.absorbing)

    # Both datasets and the actor must use actual environment action units.
    expert_actions = expert_buffer.buffer["actions"]
    if (not np.isfinite(expert_actions).all()
            or np.any(expert_actions < training_envs.action_space.low - 1e-5)
            or np.any(expert_actions > training_envs.action_space.high + 1e-5)):
        raise ValueError("Expert actions must be in environment coordinates and within its action bounds")

    algorithm: HyPE | HyPESAC

    algorithm = ALGORITHM_DICT[args.algorithm.name](training_envs=training_envs, testing_envs=None,
                                               buffer=buffer, expert_buffer=expert_buffer, agent=agent, discriminator=discriminator,
                                               logger=logger, device=device,
                                               args=args,
                                               rl_args=rl_args,
                                               discriminator_args=discriminator_args)

    logging.info("Begin Training...")
    try:
        algorithm.run()
    finally:
        training_envs.close()


if __name__ == "__main__":
    main()
