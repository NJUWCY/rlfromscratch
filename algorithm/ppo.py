import gymnasium as gym 
import torch
import math  
import numpy as np


from utils.result import Result
from utils import OPTIMIZER_DICT
from memory.memory import ReplayBuffer, TrajectoryRollout
from agent.agent import AgentBase, ProbabilityA2CAgent, DeterminiticA2CAgent
from logger.logger import Logger
from .baseonpolicy import OnPolicyAlgorithm
from utils.utils import RunningMeanStd


class PPO(OnPolicyAlgorithm):
    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: ProbabilityA2CAgent, logger: Logger, device, save_pth: str, best_pth:str, args):
        super(PPO,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth, best_pth, args)

        algo_args = args.algorithm
        self.eps_clip = algo_args.eps_clip
        self.value_coef = algo_args.value_coef
        self.use_entropy_loss= algo_args.use_entropy_loss
        self.entropy_coef = algo_args.entropy_coef
        self.actor_lr = algo_args.actor_lr
        self.critic_lr = algo_args.critic_lr 
        if algo_args.common_head:
            self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](
            [   {"params": self.agent.actor.encoder.parameters(), "lr": algo_args.encoder_lr},
                {"params": self.agent.actor.fc.parameters(), "lr": self.actor_lr},
                {"params": self.agent.critic.fc.parameters(), "lr": self.critic_lr}
                ]
            )
        else:
            self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](
                [
                    {"params": self.agent.actor.parameters(), "lr": self.actor_lr},
                    {"params": self.agent.critic.parameters(), "lr": self.critic_lr}
                    ]
                )
        self.lr_decay = algo_args.lr_decay
        if self.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(1.0 - step / self.total_update_steps, 0.0)
            )

        self.update_epochs = algo_args.update_epochs
        self.minibatch_size = algo_args.minibatch_size
        self.advan_norm = algo_args.advan_norm
        
        self.use_grad_clip = algo_args.use_grad_clip
        self.max_grad_norm = algo_args.max_grad_norm
        self.use_value_clip = algo_args.use_value_clip
        self.value_clip = algo_args.value_clip
        self.log_clip_stabilize = algo_args.log_clip_stabilize
        

    def _update_buffer(self, batch):
        self.buffer.add(batch)

    def to_correct_device_tensor(self, input, device, dtype=torch.float32)-> torch.Tensor:
        if isinstance(input, np.ndarray):
            return torch.tensor(input,dtype=dtype,device=device)
        elif isinstance(input, torch.Tensor):
            input = input.to(device)
            return input
        else:
            raise TypeError("input must be np array or torch tensor")

    def _update_with_minibatch(self,states: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor, old_values: torch.Tensor):
        """
        states: torch.Tensor:(batch, state_dim)
        actions: torch.Tensor:(batch, action_dim)
        old_log_probs: torch.Tensor:(batch,)
        advantages: torch.Tensor:(batch,)
        returns: torch.Tensor:(batch,)
        old_values: torch.Tensor:(batch,)
        """
        result_dict = {}

        log_probs = self.agent.log_prob(states,actions)

        if self.log_clip_stabilize:
            # the max clamp here is to avoid inf in ratios and advantages, which will cause the nan in gradient
            ratios = (torch.clamp(log_probs - old_log_probs,max=50)).exp()
        else:
            ratios = (log_probs - old_log_probs).exp()

        surr1 = ratios*advantages  
        surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
        actor_loss = -torch.min(surr1,surr2).mean()

        


        values = self.agent.get_value(states).squeeze(1)
        if self.use_value_clip:
            value_pred_clipped = old_values + \
                (values - old_values).clamp(-self.value_clip, self.value_clip)
            value_losses = (values - returns)**2
            value_losses_clipped = (
                value_pred_clipped - returns)**2
            critic_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            critic_loss = torch.mean((values - returns)**2)
        
        total_loss = actor_loss + self.value_coef*critic_loss 
        if self.use_entropy_loss:
            entropy = self.agent.get_entropy(states)

            total_loss -= self.entropy_coef*entropy
            result_dict['actor/entropy'] = entropy.item()
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(list(self.agent.actor.parameters()) + list(self.agent.critic.parameters()), max_norm=self.max_grad_norm)
        self.optimizer.step()
        result_dict['actor/actor_loss'] = actor_loss.item()
        result_dict['critic/loss'] = critic_loss.item()
        
        result_dict['value_mean'] = values.mean().item()
        result_dict['value_max'] = values.max().item()
        result_dict['advantage_max'] = advantages.max().item()
        result_dict['advantage_min'] = advantages.min().item()
        result_dict['ratio_max'] = ratios.max().item()
        result_dict['ratio_min'] = ratios.min().item()
        result_dict['ratio_mean'] = ratios.mean().item()


        return result_dict
       




    
    def _update_policy(self):
        with Result("train") as result:
            # get transitions
            if self.collect_traj:
                states, actions, masks, old_log_probs = self.traj_rollout.states, self.traj_rollout.actions, self.traj_rollout.masks, self.traj_rollout.log_probs
                returns, advantages, old_values = self.compute_advantages_from_traj()
            
                states = states.reshape((-1,*states.shape[2:]))
                actions = actions.reshape((-1,*actions.shape[2:]))
                masks = masks.reshape((-1))
                old_log_probs = old_log_probs.reshape((-1))
                advantages = advantages.reshape((-1))
                returns = returns.reshape((-1))
                old_values = old_values.reshape((-1))

                # remove the padding
                masks = masks==1
                states = states[masks]
                actions = actions[masks]
                old_log_probs = old_log_probs[masks]
                advantages = advantages[masks]
                returns = returns[masks]
                old_values = old_values[masks]
            else:
                states, actions, old_log_probs = self.buffer.buffer['states'], self.buffer.buffer['actions'], self.buffer.buffer['log_probs']

                returns, advantages, old_values = self.compute_advantages_from_rollout()
                states = states.reshape((-1,*states.shape[2:]))
                actions = actions.reshape((-1,*actions.shape[2:]))
                old_log_probs = old_log_probs.reshape((-1))
                advantages = advantages.reshape((-1))
                returns = returns.reshape((-1))
                old_values = old_values.reshape((-1))
                    



            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            old_log_probs = torch.from_numpy(old_log_probs).float().to(self.device)
            advantages = torch.from_numpy(advantages).float().to(self.device)
            returns = torch.from_numpy(returns).float().to(self.device)
            old_values = torch.from_numpy(old_values).float().to(self.device)

            if self.advan_norm:
                advantages = (advantages-advantages.mean())/(advantages.std()+self._eps)

            batch_size = states.shape[0]
            # here use the mini-batch update 
            
            for _ in range(self.update_epochs):
                perm_idx = torch.randperm(batch_size, device=states.device)
                for i in range(math.ceil(batch_size/self.minibatch_size)):
                    idx = perm_idx[i*self.minibatch_size:min((i+1)*self.minibatch_size,batch_size)]
                    batch_states = states[idx] 
                    batch_actions = actions[idx]
                    batch_old_log_probs = old_log_probs[idx]
                    batch_advantages = advantages[idx]
                    batch_returns = returns[idx]
                    batch_old_values = old_values[idx]
                    result_dict = self._update_with_minibatch(batch_states, batch_actions,batch_old_log_probs, batch_advantages, batch_returns, batch_old_values)
            

            if self.lr_decay:
                self.scheduler.step()


        for k,v in result_dict.items():
            result.add_metric(k, v)

        result.add_metric("actor/surrogate_target", -result_dict['actor/actor_loss'])

        if not self.advan_norm:
            result.add_metric("actor/adv_mean", advantages.mean().item())
            result.add_metric("actor/adv_max", torch.max(advantages).item())
        

        self.gradient_step += 1

        return result