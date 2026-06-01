import gymnasium as gym 
import torch
import math  
import numpy as np


from utils.result import Result
from utils import OPTIMIZER_DICT
from memory.memory import ReplayBuffer, TrajectoryRollout
from agent.agent import AgentBase
from logger.logger import Logger
from .baseonpolicy import OnPolicyAlgorithm

class PPO(OnPolicyAlgorithm):
    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: AgentBase, logger: Logger, device, save_pth: str, best_pth:str, args):
        super(PPO,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth, best_pth, args)

        algo_args = args.algorithm
        self.eps_clip = algo_args.eps_clip
        self.value_coef = algo_args.value_coef
        self.use_entropy_loss= algo_args.use_entropy_loss
        self.entropy_coef = algo_args.entropy_coef
        self.actor_lr = algo_args.actor_lr
        self.critic_lr = algo_args.critic_lr 
        self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](
            [
                {"params": self.agent.actor.parameters(), "lr": self.actor_lr},
                {"params": self.agent.critic.parameters(), "lr": self.critic_lr}
                ]
            )
        self.update_epochs = algo_args.update_epochs
        self.minibatch_size = algo_args.minibatch_size
        self.advan_norm = algo_args.advan_norm
        self.advan_eps = 1e-8
        self.use_grad_clip = algo_args.use_grad_clip
        self.max_grad_norm = algo_args.max_grad_norm

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

    def _update_with_minibatch(self,states, actions, old_log_probs, advantages, returns):
        result_dict = {}

        
        log_prob = self.agent.log_prob(states,actions)

        ratios = (log_prob - old_log_probs).exp()

        surr1 = ratios*advantages  
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*advantages
        actor_loss = -torch.min(surr1,surr2).mean()


        value = self.agent.get_value(states).squeeze(1)
        critic_loss = torch.mean((value - returns)**2)
        
        
        total_loss = actor_loss + self.value_coef*critic_loss 
        if self.use_entropy_loss:
            # since the tanh transformation makes the entropy calculation have no closed form, we use the base_dist as the entropy
            entropy = self.agent.dist(states).base_dist.entropy().mean()
            total_loss -= self.entropy_coef*entropy
            result_dict['actor/entropy'] = entropy.item()
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(list(self.agent.actor.parameters()) + list(self.agent.critic.parameters()), max_norm=self.max_grad_norm)
        self.optimizer.step()
        result_dict['actor/actor_loss'] = actor_loss.item()
        result_dict['critic/loss'] = critic_loss.item()
        
        result_dict['value_mean'] = value.mean().item()
        result_dict['value_max'] = value.max().item()
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
                returns, advantages = self.compute_advantages_from_traj()
            
                states = states.reshape((-1,states.shape[-1]))
                actions = actions.reshape((-1,actions.shape[-1]))
                masks = masks.reshape((-1))
                old_log_probs = old_log_probs.reshape((-1))
                advantages = advantages.reshape((-1))
                returns = returns.reshape((-1))

                # remove the padding
                masks = masks==1
                states = states[masks]
                actions = actions[masks]
                old_log_probs = old_log_probs[masks]
                advantages = advantages[masks]
                returns = returns[masks]
            else:
                states, actions, old_log_probs = self.buffer.buffer['states'], self.buffer.buffer['actions'], self.buffer.buffer['log_probs']

                returns, advantages = self.compute_advantages_from_rollout()
                states = states.reshape((-1,states.shape[-1]))
                actions = actions.reshape((-1,actions.shape[-1]))
                old_log_probs = old_log_probs.reshape((-1))
                advantages = advantages.reshape((-1))
                returns = returns.reshape((-1))



            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            old_log_probs = torch.from_numpy(old_log_probs).float().to(self.device)
            advantages = torch.from_numpy(advantages).float().to(self.device)
            returns = torch.from_numpy(returns).float().to(self.device)

            if self.advan_norm:
                advantages = (advantages-advantages.mean())/(advantages.std()+self.advan_eps)

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
                    result_dict = self._update_with_minibatch(batch_states, batch_actions,batch_old_log_probs, batch_advantages, batch_returns)
                    


        for k,v in result_dict.items():
            result.add_metric(k, v)

        result.add_metric("actor/surrogate_target", -result_dict['actor/actor_loss'])

        if not self.advan_norm:
            result.add_metric("actor/adv_mean", advantages.mean().item())
            result.add_metric("actor/adv_max", torch.max(advantages).item())
        
        return result