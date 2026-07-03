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
from algorithm import OffPolicyAlgorithm, DQN, ALGORITHM_DICT
from utils.utils import get_best_device, set_seed
from logger.logger import Logger 
from memory import ReplayBuffer, PrioritizedReplayBuffer
from agent.agent import AtariDQNAgent
from utils.networks import AtariDQNNetwork


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
    logger = Logger(project_name=args.experiment_name, run_name=runtime, config=args.algorithm, log_dir=args.log_dir, use_wandb=args.use_wandb, use_tensorboard=args.use_tensorboard, use_swanlab=args.use_swanlab)

    logging.info("Creating the ReplayBuffer...")
    if args.algorithm.buffer_name=="ReplayBuffer":
        buffer = ReplayBuffer(training_envs.observation_space, 
                              training_envs.action_space, 
                              args.algorithm.buffer_size//training_envs.num_envs, 
                              training_envs.num_envs,
                              gamma=args.gamma,
                              nstep=args.algorithm.nstep)
    elif args.algorithm.buffer_name=="PrioritizedReplayBuffer":
        buffer = PrioritizedReplayBuffer(training_envs.observation_space, 
                                         training_envs.action_space, 
                                         args.algorithm.buffer_size//training_envs.num_envs, 
                                         training_envs.num_envs,
                                         alpha=args.algorithm.alpha,
                                         beta=args.algorithm.beta,
                                         batch_norm=args.algorithm.weight_batch_norm,
                                         gamma=args.gamma,
                                         nstep=args.algorithm.nstep)
    logging.info("Creating the Agent...")
    network = AtariDQNNetwork(training_envs.observation_space.shape, 
                              training_envs.action_space.n, 
                              dueling_network=args.algorithm.dueling_network, 
                              conv_gradient_rescale=args.algorithm.conv_gradient_rescale)
    if args.algorithm.use_target:
        target_network = AtariDQNNetwork(training_envs.observation_space.shape, 
                                         training_envs.action_space.n, 
                                         dueling_network=args.algorithm.dueling_network,
                                         conv_gradient_rescale=args.algorithm.conv_gradient_rescale)
        
        agent = AtariDQNAgent(training_envs.observation_space, 
                              training_envs.action_space, 
                              device, 
                              network=network,
                              use_target=True,
                              target_network=target_network, 
                              target_update_tau=args.algorithm.target_update_tau,
                              target_update_method=args.algorithm.target_update_method)
    else:
        agent = AtariDQNAgent(training_envs.observation_space,
                            training_envs.action_space, 
                            device, 
                            network=network)
    
    

    # create trainer 
    algorithm: OffPolicyAlgorithm | DQN
    
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