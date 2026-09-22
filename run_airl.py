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
import numpy as np

from env.make_envs import make_vec_envs
from algorithm import ALGORITHM_DICT, AIRL, AIRLPPO, AIRLTRPO
from utils.utils import get_best_device, set_seed
from logger.logger import Logger 
from memory import ReplayBuffer, TrajectoryRollout, ExpertDataset
from agent import ProbabilityA2CAgent
from utils.networks import DiagGaussianActor, TanhGaussianActor, ValueFunction, MLPNetwork, ShapedRewardNet
from discriminator.discriminator import AIRLDiscriminator
from utils import ACTIVATION_DICT


def get_args(cfg: DictConfig):
    
    cfg.hydra_base_dir = os.getcwd()
    print("Parameters:", OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(config_path="config", config_name="config",version_base="1.3")
def main(cfg: DictConfig):
    args = get_args(cfg)
    
    set_seed(args.seed)
    
    training_envs = make_vec_envs(args.env,True,seed=args.seed) # , make_vec_envs(args.env,False,scale=False)
    
    logging.info("Checking for available GPUs...")
    device = get_best_device()

    logging.info("Creating the Logger...")
    runtime = datetime.now()
    runtime = runtime.strftime("%Y-%m-%d %H:%M:%S")
    logger = Logger(project_name=args.experiment_name, run_name=runtime, config=args, log_dir=args.log_dir, use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard, use_swanlab=args.use_swanlab)

    rl_args = args.algorithm.rl
    logging.info("Creating the ReplayBuffer...")
    if rl_args.buffer_name=="TrajectoryRollout":
        buffer = TrajectoryRollout(training_envs.observation_space,training_envs.action_space, rl_args.trajnum, args.env.max_episode_length, store_u=rl_args.rescale)
    else:
        buffer = ReplayBuffer(training_envs.observation_space, training_envs.action_space, args.interact_per_epoch, training_envs.num_envs,onpolicy=rl_args.onpolicy, store_u=rl_args.rescale)


    
    logging.info("Creating the Agent...")
    if not rl_args.rescale:
        actor = DiagGaussianActor(
            training_envs.observation_space, 
            training_envs.action_space, 
            MLPNetwork, 
            device=device,
            state_dependent_std=rl_args.state_dependent_std,
            hidden_sizes=rl_args.hidden_sizes, 
            activation=torch.nn.Tanh,
            initialize=rl_args.initialize,
            initial_log_sigma=rl_args.initial_log_sigma
        )
    else:
        if rl_args.action_bound_method == "tanh":
            actor = TanhGaussianActor(
                training_envs.observation_space, 
                training_envs.action_space, 
                MLPNetwork, 
                device=device,
                state_dependent_std=rl_args.state_dependent_std,
                hidden_sizes=rl_args.hidden_sizes, 
                activation=torch.nn.Tanh,
                initialize=rl_args.initialize,
                initial_log_sigma=rl_args.initial_log_sigma
            )
        else:
            raise ValueError(f"Action bound method {rl_args.action_bound_method} not supported")


    critic = ValueFunction(training_envs.observation_space, 
                           MLPNetwork, 
                           hidden_sizes=rl_args.hidden_sizes, 
                           activation=torch.nn.Tanh,
                           initialize=rl_args.initialize)

    agent = ProbabilityA2CAgent(training_envs.observation_space, training_envs.action_space, device, actor, critic)
    
    discriminator_args = args.algorithm.discriminator

    state_dim = training_envs.observation_space.shape[0]
    action_dim = training_envs.action_space.shape[0]
    base_inputdim = state_dim + action_dim * discriminator_args.use_action + state_dim*discriminator_args.use_next_state + 1*discriminator_args.use_done
    discriminator_base_network = MLPNetwork(base_inputdim, 
                                1, 
                                hidden_sizes=discriminator_args.base_hidden_sizes, 
                                activation=ACTIVATION_DICT[discriminator_args.base_activation])

    potential_base_network = MLPNetwork(state_dim, 
                                1, 
                                hidden_sizes=discriminator_args.potential_hidden_sizes, 
                                activation=ACTIVATION_DICT[discriminator_args.potential_activation])
    
    discriminator_net = ShapedRewardNet(discriminator_base_network, 
                                        potential_base_network, 
                                        discriminator_args.use_action,
                                        discriminator_args.use_next_state,
                                        discriminator_args.use_done,
                                        args.gamma)
    
    discriminator = AIRLDiscriminator(training_envs.observation_space, training_envs.action_space, device, discriminator_net)

    expert_buffer = ExpertDataset(args.expert_dataset.data_path, 
                                training_envs.observation_space, 
                                training_envs.action_space, 
                                args.expert_dataset.trajectory_num, 
                                args.expert_dataset.subsample_frequency,
                                args.gamma,
                                1)


    # create trainer 
    algorithm: AIRL | AIRLPPO | AIRLTRPO
    
    
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