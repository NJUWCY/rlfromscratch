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
from .gail import GAIL
from utils.utils import RunningMeanStd




class AIRL(GAIL):
    
    """GAIL algorithm implementation."""

    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout,expert_buffer:ReplayBuffer, agent: AgentBase, discriminator:DiscriminatorBase, logger: Logger, device, args: DictConfig, rl_args: DictConfig, discriminator_args: DictConfig):
        super(AIRL, self).__init__(training_envs, testing_envs, buffer,expert_buffer, agent, discriminator, logger, device, args, rl_args, discriminator_args)
        
        
        self.disc_lr = discriminator_args.disc_lr
        self.gradient_penalty_coef = discriminator_args.gradient_penalty_coef
        self.discriminator_gradient_penalty = discriminator_args.discriminator_gradient_penalty
        self.discriminator_train_steps = discriminator_args.discriminator_train_steps

        self.discriminator_batch_size = discriminator_args.discriminator_batch_size

        self.discriminator_optimizer = OPTIMIZER_DICT[discriminator_args.discriminator_optimizer](self.discriminator.parameters(), lr=self.disc_lr)
        
        self.rew_rms = RunningMeanStd()
        self.use_rms = discriminator_args.use_rms
        self.sub_logprobs = discriminator_args.sub_logprobs
        self.update_with_logprobs = discriminator_args.update_with_logprobs

    def _update_buffer(self, batch):
        with Result("buffer") as result:
            if self.collect_traj:
                # TODO: bug need to be fixed: masks the rewards so that the rms is computed correctly
                states,actions = self.traj_rollout.states, self.traj_rollout.actions
                next_states = np.concatenate([states[:,1:], self.traj_rollout.last_states[:, None]],axis=1)
                dones = self.traj_rollout.dones
                states = states.reshape((-1,*states.shape[2:]))
                actions = actions.reshape((-1,*actions.shape[2:])) 
                next_states = next_states.reshape((-1,*next_states.shape[2:]))
                dones = dones.reshape((-1,1))
                with torch.no_grad():
                    rewards = self.discriminator.predict_reward(states,actions,next_states,dones).squeeze(-1)
                if self.use_rms:
                    self.traj_rollout.rewards = self.rew_rms.norm(rewards).reshape(*self.traj_rollout.rewards.shape)
                    self.rew_rms.update(rewards)
                else:
                    self.traj_rollout.rewards = rewards.reshape(*self.traj_rollout.rewards.shape)

            else:
                states,actions = batch['states'], batch['actions']
                next_states = batch['next_states']
                dones = batch['dones']
                log_probs = batch['log_probs']
                states = states.reshape((-1,*states.shape[2:]))
                actions = actions.reshape((-1,*actions.shape[2:])) 
                next_states = next_states.reshape((-1,*next_states.shape[2:]))
                dones = dones.reshape((-1,1))
                log_probs = log_probs.reshape((-1,1))
                with torch.no_grad():
                    rewards = self.discriminator.predict_reward(states,actions,next_states,dones,log_probs,self.sub_logprobs).squeeze(-1)
                if self.use_rms:
                    batch['rewards'] = self.rew_rms.norm(rewards).reshape(*batch['rewards'].shape)
                    self.rew_rms.update(rewards)
                else:
                    batch['rewards'] = rewards.reshape(*batch['rewards'].shape)
                self.buffer.add(batch) 
        result.add_metric("buffer/reward_mean", rewards.mean().item())
        return result
    
    def _compute_gradient_penalty(self, 
            expert_states:torch.Tensor, 
            expert_actions:torch.Tensor, 
            expert_next_states:torch.Tensor,
            expert_dones:torch.Tensor,
            policy_states:torch.Tensor, 
            policy_actions:torch.Tensor,
            policy_next_states:torch.Tensor,
            policy_dones:torch.Tensor):
        alpha = torch.rand(expert_states.shape[0],1,device=expert_states.device)
        expert_data = torch.cat([expert_states, expert_actions, expert_next_states, expert_dones], dim=1)
        policy_data = torch.cat([policy_states, policy_actions, policy_next_states, policy_dones], dim=1)
        state_dim = expert_states.shape[-1]
        action_dim = expert_actions.shape[-1]
        # TODO: This might be expanded to visual features
        mixed_data = alpha*expert_data + (1-alpha)*policy_data
        mixed_data.requires_grad = True
        mixed_logits = self.discriminator.predict_logits(
            mixed_data[:,:state_dim], 
            mixed_data[:,state_dim:state_dim+action_dim], 
            mixed_data[:,state_dim+action_dim:state_dim+action_dim+state_dim], 
            mixed_data[:,state_dim+action_dim+state_dim:])

        gradient = torch.autograd.grad(outputs=mixed_logits.sum(), inputs=mixed_data, create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = self.gradient_penalty_coef*((gradient_norm - 1)**2).mean()
        return gradient_penalty

    def _update_discriminator(self, batch:dict):
        with Result("discriminator") as result:
            expert_states = self.expert_buffer.buffer['states']
            expert_actions = self.expert_buffer.buffer['actions']
            expert_next_states = self.expert_buffer.buffer['next_states']
            expert_dones = self.expert_buffer.buffer['dones']
            

            expert_states = expert_states.reshape((-1,*expert_states.shape[2:]))
            expert_actions = expert_actions.reshape((-1,*expert_actions.shape[2:]))
            expert_next_states = expert_next_states.reshape((-1,*expert_next_states.shape[2:]))
            expert_dones = expert_dones.reshape((-1,1))

            # normalize expert states
            if self.args.env.obs_norm:
                expert_states = self.training_envs._norm_obs(expert_states)
                expert_next_states = self.training_envs._norm_obs(expert_next_states)
            
            if self.collect_traj:
                policy_states = self.traj_rollout.states
                policy_actions = self.traj_rollout.actions
                policy_next_states = np.concatenate([self.traj_rollout.states[:,1:], self.traj_rollout.last_states[:, None]], axis=1)
                policy_dones = self.traj_rollout.dones
                policy_log_probs = self.traj_rollout.log_probs
                masks = self.traj_rollout.masks

                policy_states = policy_states.reshape((-1,*policy_states.shape[2:]))
                policy_actions = policy_actions.reshape((-1,*policy_actions.shape[2:]))
                policy_next_states = policy_next_states.reshape((-1,*policy_next_states.shape[2:]))
                policy_dones = policy_dones.reshape((-1,1))
                policy_log_probs = policy_log_probs.reshape((-1,1))
                masks = masks.reshape((-1,))


                masks = masks==1
                policy_states = policy_states[masks]
                policy_actions = policy_actions[masks]
                policy_next_states = policy_next_states[masks]
                policy_dones = policy_dones[masks]
                policy_log_probs = policy_log_probs[masks]
                
            else:
                policy_states = batch['states']
                policy_actions = batch['actions']
                policy_next_states = batch['next_states']
                policy_dones = batch['dones']
                policy_log_probs = batch['log_probs']

                policy_states = policy_states.reshape((-1,*policy_states.shape[2:]))
                policy_actions = policy_actions.reshape((-1,*policy_actions.shape[2:]))
                policy_next_states = policy_next_states.reshape((-1,*policy_next_states.shape[2:]))
                policy_dones = policy_dones.reshape((-1,1))
                policy_log_probs = policy_log_probs.reshape((-1,1))



            # compute log_probs 
            with torch.no_grad():
                expert_log_probs = self.agent.log_prob(expert_states, expert_actions).reshape((-1,1))
                # policy_log_probs = self.agent.log_prob(policy_states, policy_actions).reshape((-1,1))
            

            # transform np arrays into torch tensors
            expert_states = torch.from_numpy(expert_states).float().to(self.device)
            expert_actions = torch.from_numpy(expert_actions).float().to(self.device)
            expert_next_states = torch.from_numpy(expert_next_states).float().to(self.device)
            expert_dones = torch.from_numpy(expert_dones).float().to(self.device)
            expert_log_probs = expert_log_probs.to(self.device)

            policy_states = torch.from_numpy(policy_states).float().to(self.device)
            policy_actions = torch.from_numpy(policy_actions).float().to(self.device)
            policy_next_states = torch.from_numpy(policy_next_states).float().to(self.device)
            policy_dones = torch.from_numpy(policy_dones).float().to(self.device)
            policy_log_probs = torch.from_numpy(policy_log_probs).float().to(self.device)

            expert_perm_idx = torch.randperm(expert_states.shape[0], device=expert_states.device)
            policy_perm_idx = torch.randperm(policy_states.shape[0], device=policy_states.device) 
            mini_steps = min(
                math.floor(expert_states.shape[0] / self.discriminator_batch_size),
                math.floor(policy_states.shape[0] / self.discriminator_batch_size),
            )
            assert self.discriminator_train_steps <= mini_steps, "discriminator_train_steps must be less than or equal to the minimum of the number of expert and policy samples"

            for i in range(self.discriminator_train_steps):
                
                expert_idx = expert_perm_idx[i * self.discriminator_batch_size:(i + 1) * self.discriminator_batch_size]
                policy_idx = policy_perm_idx[i * self.discriminator_batch_size:(i + 1) * self.discriminator_batch_size]


                expert_states_batch = expert_states[expert_idx]
                expert_actions_batch = expert_actions[expert_idx]
                expert_log_probs_batch = expert_log_probs[expert_idx]
                expert_next_states_batch = expert_next_states[expert_idx]
                expert_dones_batch = expert_dones[expert_idx]

                policy_states_batch = policy_states[policy_idx]
                policy_actions_batch = policy_actions[policy_idx]
                policy_log_probs_batch = policy_log_probs[policy_idx]
                policy_next_states_batch = policy_next_states[policy_idx]
                policy_dones_batch = policy_dones[policy_idx]
                if self.update_with_logprobs:
                    expert_logits = self.discriminator.predict_logits(expert_states_batch, expert_actions_batch, expert_next_states_batch, expert_dones_batch) - expert_log_probs_batch
                    policy_logits = self.discriminator.predict_logits(policy_states_batch, policy_actions_batch, policy_next_states_batch, policy_dones_batch) - policy_log_probs_batch
                else:
                    expert_logits = self.discriminator.predict_logits(expert_states_batch, expert_actions_batch, expert_next_states_batch, expert_dones_batch)
                    policy_logits = self.discriminator.predict_logits(policy_states_batch, policy_actions_batch, policy_next_states_batch, policy_dones_batch)
                # loss = torch.nn.functional.binary_cross_entropy_with_logits(expert_logits, torch.zeros_like(expert_logits)) + torch.nn.functional.binary_cross_entropy_with_logits(policy_logits, torch.ones_like(policy_logits))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits)) + torch.nn.functional.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))

                if self.discriminator_gradient_penalty:
                    loss += self._compute_gradient_penalty(expert_states_batch, 
                                                           expert_actions_batch, 
                                                           expert_next_states_batch,
                                                           expert_dones_batch,
                                                           policy_states_batch, 
                                                           policy_actions_batch,
                                                           policy_next_states_batch,
                                                           policy_dones_batch)

                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()

        result.add_metric("discriminator/loss", loss.item())

        # Recompute logits on the full expert/policy sets for logging.
        with torch.no_grad():
            expert_base_logits = self.discriminator.predict_logits(
                expert_states, expert_actions, expert_next_states, expert_dones
            )
            policy_base_logits = self.discriminator.predict_logits(
                policy_states, policy_actions, policy_next_states, policy_dones
            )
            if self.update_with_logprobs:
                expert_logits = expert_base_logits - expert_log_probs
                policy_logits = policy_base_logits - policy_log_probs
            else:
                expert_logits = expert_base_logits
                policy_logits = policy_base_logits

        result.add_metric("discriminator/expert_D", nn.functional.sigmoid(expert_logits).mean().item())
        result.add_metric("discriminator/policy_D", nn.functional.sigmoid(policy_logits).mean().item())
        result.add_metric("discriminator/expert_base_logits", expert_base_logits.mean().item())
        result.add_metric("discriminator/policy_base_logits", policy_base_logits.mean().item())
        # dL/df_i = (sigmoid(f_i - log_pi_i) - 1) / N, since the expert term of the loss is
        # mean_i softplus(-(f_i - log_pi_i)) and log_pi_i is a constant w.r.t. the discriminator.
        expert_logits_grad = (nn.functional.sigmoid(expert_logits) - 1.0) / expert_logits.numel()
        result.add_metric("discriminator/expert_logits_grad_mean", expert_logits_grad.abs().mean().item())
        policy_logits_grad = nn.functional.sigmoid(policy_logits) / policy_logits.numel()
        result.add_metric("discriminator/policy_logits_grad_mean", policy_logits_grad.abs().mean().item())
        result.add_metric("discriminator/expert_log_probs", expert_log_probs.mean().item())
        result.add_metric("discriminator/policy_log_probs", policy_log_probs.mean().item())

        # Same discriminator diagnostics as imitation.algorithms.adversarial.common.compute_train_stats.
        with torch.no_grad():
            disc_logits = torch.cat([expert_logits.reshape(-1), policy_logits.reshape(-1)], dim=0)
            labels_expert_is_one = torch.cat(
                [
                    torch.ones(expert_logits.numel(), device=disc_logits.device),
                    torch.zeros(policy_logits.numel(), device=disc_logits.device),
                ],
                dim=0,
            )
            bin_is_generated_pred = disc_logits < 0
            bin_is_generated_true = labels_expert_is_one == 0
            bin_is_expert_true = torch.logical_not(bin_is_generated_true)
            n_generated = float(bin_is_generated_true.sum())
            n_labels = float(labels_expert_is_one.numel())
            n_expert = n_labels - n_generated
            n_expert_pred = float((~bin_is_generated_pred).sum())
            correct_vec = torch.eq(bin_is_generated_pred, bin_is_generated_true)
            disc_acc = float(correct_vec.float().mean())
            if n_expert < 1:
                disc_acc_expert = float("nan")
            else:
                disc_acc_expert = float(
                    torch.logical_and(bin_is_expert_true, correct_vec).sum().item() / n_expert
                )
            disc_acc_gen = float(
                torch.logical_and(bin_is_generated_true, correct_vec).sum().item()
                / max(1.0, n_generated)
            )
            disc_entropy = float(torch.distributions.Bernoulli(logits=disc_logits).entropy().mean())
            disc_proportion_expert_true = n_expert / n_labels if n_labels > 0 else float("nan")
            disc_proportion_expert_pred = n_expert_pred / n_labels if n_labels > 0 else float("nan")

        result.add_metric("discriminator/disc_acc", disc_acc)
        result.add_metric("discriminator/disc_acc_expert", disc_acc_expert)
        result.add_metric("discriminator/disc_acc_gen", disc_acc_gen)
        result.add_metric("discriminator/disc_entropy", disc_entropy)
        result.add_metric("discriminator/disc_proportion_expert_true", disc_proportion_expert_true)
        result.add_metric("discriminator/disc_proportion_expert_pred", disc_proportion_expert_pred)
        result.add_metric("discriminator/n_expert", n_expert)
        result.add_metric("discriminator/n_generated", n_generated)
        return result

class AIRLPPO(AIRL,PPO):
    pass 

class AIRLTRPO(AIRL,TRPO):
    pass 
