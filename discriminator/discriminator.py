import torch.nn as nn
import torch
from torch.distributions import Normal, TransformedDistribution, Independent
from torch.distributions.transforms import AffineTransform, TanhTransform
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Space 
from typing import Union, Tuple, Optional, Dict
from gymnasium.spaces import Box, Discrete

from utils.utils import RunningMeanStd

def to_correct_device_tensor(input, device, dtype=torch.float32)-> torch.Tensor:
    if isinstance(input, np.ndarray):
        return torch.tensor(input,dtype=dtype,device=device)
    elif isinstance(input, torch.Tensor):
        input = input.to(device)
        return input
    else:
        raise TypeError("input must be np array or torch tensor")

class DiscriminatorBase(nn.Module, ABC):
    # config_attrs is a tuple of strings that are the names of the attributes that are used to configure the agent
    _config_attrs: tuple[str, ...] = ("observation_space", "action_space") 
    def __init__(self,observation_space: Space, action_space: Space, device:torch.device):
        super(DiscriminatorBase, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space 
        self.device = device
        
    
    @abstractmethod
    def predict_reward(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        """Predict the reward for the given states and actions. """
        pass 
    
   
    def _get_config_dict(self) -> dict:
        return {name: getattr(self, name) for name in self._config_attrs}


    def save(self, pth_file: str):
        """
        Save model to a given location.

        :param path:
        """
        # we need to save the states_dict and other parameters that are used to configure the agent, 
        # but notice: 1. device should not be saved 2. the extra parameters in actor, critic are not saved. These parameters should be appointed by the user when creating the model.
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self._get_config_dict()
            },
            pth_file
        )
        
    def load(self,pth_file:str):
        ckpt = torch.load(pth_file, weights_only=False, map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        for name, value in ckpt["config"].items():
            setattr(self, name, value)



# TODO: add visual input support for discriminator, currently only support vector input
REWARD_FUNCTIONS = ("GAIL", "AIRL", "FAIRL")


class GAILDiscriminator(DiscriminatorBase):
    def __init__(self,observation_space: Space, action_space: Space, device:torch.device, network:nn.Module, reward_function:str="GAIL"):
        super(GAILDiscriminator, self).__init__(observation_space, action_space, device)
        self.input_dim = observation_space.shape[0] + action_space.shape[0]
        self.network = network
        if reward_function not in REWARD_FUNCTIONS:
            raise ValueError(f"reward_function must be one of {REWARD_FUNCTIONS}, got {reward_function}")
        self.reward_function = reward_function
    
    def predict_reward(self, states:Union[np.ndarray, torch.Tensor], actions:Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        states = to_correct_device_tensor(states, self.device)
        actions = to_correct_device_tensor(actions, self.device)
        assert states.shape[1] == self.observation_space.shape[0]
        inputs = torch.concat([states, actions], axis=1)
        logits = self.network(inputs)
        if self.reward_function == "GAIL":
            D = nn.functional.sigmoid(logits)
            # Strictly positive survival bonus, the usual choice for DAC.
            rewards = -torch.log(1-D)
        else:
            # log D - log(1 - D)=logits, i.e. the discriminator logits.
            rewards = logits
            if self.reward_function == "FAIRL":
                rewards = torch.exp(rewards) * -rewards
        rewards = rewards.detach().cpu().numpy()

        return rewards
    
    def predict_logits(self, states_actions:Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        states_actions = to_correct_device_tensor(states_actions, self.device)
        assert states_actions.shape[1] == self.input_dim
        logits = self.network(states_actions)
        return logits


class AIRLDiscriminator(DiscriminatorBase):
    def __init__(self,observation_space: Space, action_space: Space, device:torch.device, network:nn.Module):
        super(AIRLDiscriminator, self).__init__(observation_space, action_space, device)
        self.input_dim = observation_space.shape[0] + action_space.shape[0]
        self.network = network
    
    def predict_reward(self, states:Union[np.ndarray, torch.Tensor], 
            actions:Union[np.ndarray, torch.Tensor], 
            next_states:Union[np.ndarray, torch.Tensor], 
            dones:Union[np.ndarray, torch.Tensor], 
            log_probs:Union[np.ndarray, torch.Tensor], 
            sub_logprobs:bool) -> np.ndarray:

        states = to_correct_device_tensor(states, self.device)
        actions = to_correct_device_tensor(actions, self.device)
        next_states = to_correct_device_tensor(next_states, self.device)
        dones = to_correct_device_tensor(dones, self.device)
        log_probs = to_correct_device_tensor(log_probs, self.device)
        assert states.shape[1] == self.observation_space.shape[0]
        if sub_logprobs:
            rewards = self.network(states, actions, next_states, dones) - log_probs
        else:
            rewards = self.network(states, actions, next_states, dones)
        
        
        rewards = rewards.detach().cpu().numpy()
        return rewards 
    
    
    def predict_logits(self, states:Union[np.ndarray, torch.Tensor], actions:Union[np.ndarray, torch.Tensor], next_states:Union[np.ndarray, torch.Tensor], dones:Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        states = to_correct_device_tensor(states, self.device)
        actions = to_correct_device_tensor(actions, self.device)
        next_states = to_correct_device_tensor(next_states, self.device)
        dones = to_correct_device_tensor(dones, self.device)
        assert states.shape[1] == self.observation_space.shape[0]
        logits = self.network(states, actions, next_states, dones)
        return logits
