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
from datetime import datetime 

from env.make_envs import make_vec_envs
from algorithm import ALGORITHM_DICT, DAC, DACSAC
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

@hydra.main(config_path="config", config_name="config",version_base="1.3")
def main(cfg: DictConfig):
    args = get_args(cfg)

    set_seed(args.seed)

    training_envs = make_vec_envs(args.env,True,seed=args.seed)

    logging.info("Checking for available GPUs...")
    device = get_best_device()

    logging.info("Creating the Logger...")
    runtime = datetime.now()
    runtime = runtime.strftime("%Y-%m-%d %H:%M:%S")
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
    if rl_args.action_bound_method != "tanh":
        raise ValueError(f"DAC requires a squashed Gaussian policy, got action_bound_method={rl_args.action_bound_method}")
    actor = TanhGaussianActor(
        training_envs.observation_space,
        training_envs.action_space,
        MLPNetwork,
        device=device,
        state_dependent_std=rl_args.state_dependent_std,
        hidden_sizes=rl_args.hidden_sizes,
        activation=activation,
    )

    critic_class = DoubleQFunction if rl_args.double_critic else QFunction
    critic = critic_class(training_envs.observation_space, training_envs.action_space, MLPNetwork, hidden_sizes=rl_args.hidden_sizes, activation=activation)
    target_critic = critic_class(training_envs.observation_space, training_envs.action_space, MLPNetwork, hidden_sizes=rl_args.hidden_sizes, activation=activation) if rl_args.use_target else None

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
    discriminator_net = MLPNetwork(training_envs.observation_space.shape[0]+training_envs.action_space.shape[0],
                                   1,
                                   hidden_sizes=discriminator_args.discriminator_hidden_sizes,
                                   activation=ACTIVATION_DICT[discriminator_args.discriminator_activation])

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

    algorithm: DAC | DACSAC

    algorithm = ALGORITHM_DICT[args.algorithm.name](training_envs=training_envs, testing_envs=None,
                                               buffer=buffer, expert_buffer=expert_buffer, agent=agent, discriminator=discriminator,
                                               logger=logger, device=device,
                                               args=args,
                                               rl_args=rl_args,
                                               discriminator_args=discriminator_args)

    logging.info("Begin Training...")
    algorithm.run()


if __name__ == "__main__":
    main()
