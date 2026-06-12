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
from algorithm import TRPO, OnPolicyAlgorithm, ALGORITHM_DICT, PPO
from utils.utils import get_best_device, set_seed
from logger.logger import Logger 
from memory import BUFFER_DICT,ReplayBuffer, TrajectoryRollout
from agent import ProbabilityA2CAgent
from utils.networks import DiagGaussianActor, TanhGaussianActor, ValueFunction, MLPNetwork, AtariCNNEncoder, DiscreteProbabilityActor, DiscreteVFunction


def get_args(cfg: DictConfig):
    
    cfg.hydra_base_dir = os.getcwd()
    print("Parameters:", OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(config_path="config", config_name="config",version_base="1.3")
def main(cfg: DictConfig):
    args = get_args(cfg)
    set_seed(args.seed)
    
    # Attention: here we set the scale=True in PPO training.
    training_envs = make_vec_envs(args.env,True,seed=args.seed) # , make_vec_envs(args.env,False,scale=False)
    
    logging.info("Checking for available GPUs...")
    device = get_best_device()

    logging.info("Creating the Logger...")
    runtime = datetime.now()
    runtime = runtime.strftime("%Y-%m-%d %H:%M:%S")
    logger = Logger(project_name=args.experiment_name, run_name=runtime, config=args.algorithm, log_dir=args.log_dir, use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard, use_swanlab=args.use_swanlab)

    logging.info("Creating the ReplayBuffer...")
    if args.algorithm.buffer_name=="TrajectoryRollout":
        buffer = TrajectoryRollout(training_envs.observation_space,training_envs.action_space, args.algorithm.trajnum, args.env.max_episode_length)
    else:
        buffer = ReplayBuffer(training_envs.observation_space, training_envs.action_space, args.interact_per_epoch, training_envs.num_envs,onpolicy=args.algorithm.onpolicy)

    logging.info("Creating the Agent...")
    

    embedding_dim = 512 
    encoder = AtariCNNEncoder(
        training_envs.observation_space.shape, 
        embedding_dim, 
        initialize=args.algorithm.initialize)

    actor = DiscreteProbabilityActor(
        training_envs.observation_space, 
        training_envs.action_space, 
        encoder,
        initialize=args.algorithm.initialize)
    
    critic = DiscreteVFunction(
        training_envs.observation_space, 
        training_envs.action_space, 
        encoder,
        initialize=args.algorithm.initialize
    )
    
    # TODO: actor and critic share the encoder, which may cause repetition in agent.state_dict()
    agent = ProbabilityA2CAgent(training_envs.observation_space, training_envs.action_space, device, actor, critic)
    
        

    # create trainer 
    algorithm: OnPolicyAlgorithm | PPO
    
    algorithm = ALGORITHM_DICT[args.algorithm.name](training_envs=training_envs, testing_envs=None, 
                                               buffer=buffer, agent=agent, 
                                               logger=logger, device=device, 
                                               save_pth=os.path.join(args.log_dir, "newest_model.pth"), 
                                               best_pth=os.path.join(args.log_dir, "best_model.pth"),
                                               args=args)

    logging.info("Begin Training...")
    algorithm.run()
    

if __name__ == "__main__":
    main()