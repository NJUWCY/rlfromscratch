from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import gymnasium as gym
import torch 
import math
import torch.nn as nn
from omegaconf import DictConfig

from .baseailalgorithm import AILAlgorithm
from .baseonpolicy import OnPolicyAlgorithm
from .ppo import PPO
from .trpo import TRPO
from memory.memory import ReplayBuffer,TrajectoryRollout
from utils.result import Result
from logger.logger import Logger
from agent.agent import AgentBase
from discriminator.discriminator import DiscriminatorBase
from utils import OPTIMIZER_DICT




class GAIL(AILAlgorithm,OnPolicyAlgorithm):
    
    """GAIL algorithm implementation."""

    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout,expert_buffer:ReplayBuffer, agent: AgentBase, discriminator:DiscriminatorBase, logger: Logger, device, args: DictConfig, rl_args: DictConfig, discriminator_args: DictConfig):
        super(GAIL, self).__init__(training_envs, testing_envs, buffer,expert_buffer, agent, discriminator, logger, device, args, rl_args, discriminator_args)
        
        
        self.disc_lr = discriminator_args.disc_lr
        self.gradient_penalty_coef = discriminator_args.gradient_penalty_coef
        self.discriminator_gradient_penalty = discriminator_args.discriminator_gradient_penalty
        self.discriminator_train_steps = discriminator_args.discriminator_train_steps
        # self.discriminator_train_epochs = (discriminator_args.discriminator_train_epochs)
        self.discriminator_batch_size = discriminator_args.discriminator_batch_size

        self.discriminator_optimizer = OPTIMIZER_DICT[discriminator_args.discriminator_optimizer](self.discriminator.parameters(), lr=self.disc_lr)
        


    def update(self, batch, start_train)-> Result:
        if start_train:
            for _ in range(self.update_discriminator_step_per_epoch):
                discriminator_result = self._update_discriminator(batch)
          

        result = self._update_buffer(batch)
            
        if start_train:
            result.add(discriminator_result)
        
        if start_train:
            for _ in range(self.update_step_per_epoch):
                update_policy_log = self._update_policy()
            result.add(update_policy_log)
        
        if self.collect_traj:
            self.traj_rollout.reset()
        else:
            self.buffer.reset()
        return result
        

    def _update_buffer(self, batch):
        with Result("buffer") as result:
            if self.collect_traj:
                states,actions = self.traj_rollout.states, self.traj_rollout.actions
                states = states.reshape((-1,*states.shape[2:]))
                actions = actions.reshape((-1,*actions.shape[2:])) 
                with torch.no_grad():
                    rewards = self.discriminator.predict_reward(states,actions).squeeze(-1)
                self.traj_rollout.rewards = rewards.reshape(*self.traj_rollout.rewards.shape)
            else:
                states,actions = batch['states'], batch['actions']
                states = states.reshape((-1,*states.shape[2:]))
                actions = actions.reshape((-1,*actions.shape[2:])) 
                with torch.no_grad():
                    rewards = self.discriminator.predict_reward(states,actions).squeeze(-1)
                batch['rewards'] = rewards.reshape(*batch['rewards'].shape)
                self.buffer.add(batch) 
        result.add_metric("discriminator_rewards", rewards.mean().item())
        return result
        
    def _compute_gradient_penalty(self, expert_states:torch.Tensor, expert_actions:torch.Tensor, policy_states:torch.Tensor, policy_actions:torch.Tensor):
        alpha = torch.rand(expert_states.shape[0],1,device=expert_states.device)
        expert_data = torch.cat([expert_states, expert_actions], dim=1)
        policy_data = torch.cat([policy_states, policy_actions], dim=1)
        # TODO: This might be expanded to visual features
        mixed_data = alpha*expert_data + (1-alpha)*policy_data
        mixed_data.requires_grad = True
        mixed_logits = self.discriminator.predict_logits(mixed_data)

        gradient = torch.autograd.grad(outputs=mixed_logits.sum(), inputs=mixed_data, create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = self.gradient_penalty_coef*((gradient_norm - 1)**2).mean()
        return gradient_penalty

    def _update_discriminator(self, batch:dict):
        with Result("discriminator") as result:
            expert_states = self.expert_buffer.buffer['states']
            expert_actions = self.expert_buffer.buffer['actions']
            expert_states = expert_states.reshape((-1,*expert_states.shape[2:]))
            expert_actions = expert_actions.reshape((-1,*expert_actions.shape[2:]))

            # normalize expert states
            if self.args.env.obs_norm:
                expert_states = self.training_envs._norm_obs(expert_states)
            
            if self.collect_traj:
                policy_states = self.traj_rollout.states
                policy_actions = self.traj_rollout.actions
                masks = self.traj_rollout.masks 
                policy_states = policy_states.reshape((-1,*policy_states.shape[2:]))
                policy_actions = policy_actions.reshape((-1,*policy_actions.shape[2:]))
                masks = masks.reshape((-1,))
                masks = masks==1
                policy_states = policy_states[masks]
                policy_actions = policy_actions[masks]
            
            else:
                policy_states = batch['states']
                policy_actions = batch['actions']
                policy_states = policy_states.reshape((-1,*policy_states.shape[2:]))
                policy_actions = policy_actions.reshape((-1,*policy_actions.shape[2:]))
            
            expert_states = torch.from_numpy(expert_states).float().to(self.device)
            expert_actions = torch.from_numpy(expert_actions).float().to(self.device)
            policy_states = torch.from_numpy(policy_states).float().to(self.device)
            policy_actions = torch.from_numpy(policy_actions).float().to(self.device)
            
            expert_perm_idx = torch.randperm(expert_states.shape[0], device=expert_states.device)
            policy_perm_idx = torch.randperm(policy_states.shape[0], device=policy_states.device) 
            mini_steps = min(
                math.floor(expert_states.shape[0] / self.discriminator_batch_size),
                math.floor(policy_states.shape[0] / self.discriminator_batch_size),
            )
            assert self.discriminator_train_steps <= mini_steps, "discriminator_train_steps must be less than or equal to the minimum of the number of expert and policy samples"
            # for _ in range(self.discriminator_train_steps): 
            #     expert_idx = torch.randint(0, expert_states.shape[0], (self.discriminator_expert_batch_size,), device=expert_states.device)
            #     policy_idx = torch.randint(0, policy_states.shape[0], (self.discriminator_policy_batch_size,), device=policy_states.device)
                
            for i in range(self.discriminator_train_steps):
                
                expert_idx = expert_perm_idx[i * self.discriminator_batch_size:(i + 1) * self.discriminator_batch_size]
                policy_idx = policy_perm_idx[i * self.discriminator_batch_size:(i + 1) * self.discriminator_batch_size]


                expert_states_batch = expert_states[expert_idx]
                expert_actions_batch = expert_actions[expert_idx]
                policy_states_batch = policy_states[policy_idx]
                policy_actions_batch = policy_actions[policy_idx]
                
                expert_logits = self.discriminator.predict_logits(torch.cat([expert_states_batch, expert_actions_batch], dim=1))
                policy_logits = self.discriminator.predict_logits(torch.cat([policy_states_batch, policy_actions_batch], dim=1))

                # loss = torch.nn.functional.binary_cross_entropy_with_logits(expert_logits, torch.zeros_like(expert_logits)) + torch.nn.functional.binary_cross_entropy_with_logits(policy_logits, torch.ones_like(policy_logits))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits)) + torch.nn.functional.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))

                if self.discriminator_gradient_penalty:
                    loss += self._compute_gradient_penalty(expert_states_batch, expert_actions_batch, policy_states_batch, policy_actions_batch)

                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()

        
        result.add_metric("discriminator/loss", loss.item())
        result.add_metric("discriminator/expert_D", nn.functional.sigmoid(expert_logits).mean().item())
        result.add_metric("discriminator/policy_D", nn.functional.sigmoid(policy_logits).mean().item())
        return result

class GAILPPO(GAIL,PPO):
    pass 

class GAILTRPO(GAIL,TRPO):
    pass 
