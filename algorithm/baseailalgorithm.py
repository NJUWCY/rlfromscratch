from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import gymnasium as gym
import torch 


from logger.logger import Logger
from memory.memory import ReplayBuffer
from agent.agent import AgentBase
from discriminator.discriminator import DiscriminatorBase
from utils.result import Result
from . import BaseAlgorithm
from typing import Dict, Any
from omegaconf import DictConfig

class AILAlgorithm(BaseAlgorithm, ABC):

    """Base class for off-policy RL algorithms."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer,expert_buffer:ReplayBuffer, agent: AgentBase, discriminator:DiscriminatorBase, logger: Logger, device, args: DictConfig, rl_args: DictConfig, discriminator_args: DictConfig):
        super(AILAlgorithm,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, args, rl_args)
        self.start_train_step = args.start_train_step
        self.discriminator = discriminator.to(self.device)
        self.expert_buffer = expert_buffer
        self.update_discriminator_step_per_epoch = args.update_discriminator_step_per_epoch
    
    
    def update(self, batch, start_train)-> Result:
        raise NotImplementedError("This method should be implemented in the subclass")
        
        
    
    @abstractmethod
    def _update_buffer(self, batch):
        """
        update the replay buffer with given selected batch of data
        :param batch: the batch of data
        """
    
    
    @abstractmethod
    def _update_discriminator(self):
        """
        do gradient update to the discriminator with a batch of data sampled from replay buffer
        """

        

    def start_train(self):
        return self.interaction_step>=self.start_train_step






