import logging
import os
from datetime import datetime

import gymnasium as gym
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from agent import PolicyAgent
from algorithm import ALGORITHM_DICT, BehaviorCloning, OfflineAlgorithm
from env.make_envs import make_vec_envs
from logger.logger import Logger
from memory import ExpertDataset
from utils import ACTIVATION_DICT
from utils.networks import GaussianActor, MLPNetwork
from utils.utils import get_best_device, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] ---------------%(message)s---------------",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_args(cfg: DictConfig):
    cfg.hydra_base_dir = os.getcwd()
    print("Parameters:", OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    args = get_args(cfg)
    if args.algorithm.name != "BC":
        raise ValueError("run_bc.py must be launched with algorithm=bc.")

    set_seed(args.seed)

    logging.info("Creating the environment for spaces and evaluation...")
    training_envs = make_vec_envs(args.env, True, scale=False, seed=args.seed)
    if not isinstance(training_envs.action_space, gym.spaces.Box):
        raise NotImplementedError("BC training currently implements only continuous Box action spaces.")

    logging.info("Checking for available GPUs...")
    device = get_best_device()

    logging.info("Creating the Logger...")
    runtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger = Logger(
        project_name=args.experiment_name,
        run_name=runtime,
        config=args.algorithm,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        use_swanlab=args.use_swanlab,
    )

    logging.info("Loading expert dataset...")
    dataset = ExpertDataset(
        args.algorithm.dataset_path,
        dataset_format=args.algorithm.dataset_format,
        state_key=args.algorithm.state_key,
        action_key=args.algorithm.action_key,
        minari_download=args.algorithm.minari_download,
        minari_datasets_path=args.algorithm.minari_datasets_path,
    )

    logging.info("Creating the policy agent...")
    activation = ACTIVATION_DICT[args.algorithm.activation]
    actor = GaussianActor(
        training_envs.observation_space,
        training_envs.action_space,
        MLPNetwork,
        device=device,
        rescale=args.algorithm.rescale,
        action_bound_method="tanh",
        hidden_sizes=args.algorithm.hidden_sizes,
        activation=activation,
    )
    agent = PolicyAgent(training_envs.observation_space, training_envs.action_space, device, actor)

    algorithm: OfflineAlgorithm | BehaviorCloning
    algorithm = ALGORITHM_DICT[args.algorithm.name](
        training_envs=training_envs,
        dataset=dataset,
        agent=agent,
        logger=logger,
        device=device,
        save_pth=os.path.join(args.log_dir, "newest_model.pth"),
        best_pth=os.path.join(args.log_dir, "best_model.pth"),
        args=args,
    )

    logging.info("Begin BC training...")
    algorithm.run()
    logger.close()
    training_envs.close()


if __name__ == "__main__":
    main()
