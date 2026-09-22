from .baseoffpolicy import OffPolicyAlgorithm
from memory.memory import ReplayBuffer
import torch 
from utils.result import Result
from logger.logger import Logger
import numpy as np
from agent import AgentBase, A2CAgent, SACAgent
from utils import OPTIMIZER_DICT
import math
from typing import Dict, Any
from omegaconf import DictConfig

class SAC(OffPolicyAlgorithm):
    
    """DQN algorithm implementation."""

    def __init__(self, training_envs, testing_envs, buffer: ReplayBuffer, agent: SACAgent, logger: Logger, device, args: DictConfig, rl_args: DictConfig):
        super(SAC, self).__init__(training_envs, testing_envs, buffer, agent, logger, device, args, rl_args)
        
        algo_args = rl_args

        self.critic_lr = algo_args.critic_lr
        self.actor_lr = algo_args.actor_lr
        self.temp_lr = algo_args.temp_lr
        
       
        self.batch_size = algo_args.batch_size
        

        self.agent = agent

        self.use_target = algo_args.use_target 
        self.target_update_interval = algo_args.target_update_interval

        
        self.critic_optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.critic_optimizer](self.agent.critic.parameters(), lr=self.critic_lr)
        self.actor_optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.actor_optimizer](self.agent.actor.parameters(), lr=self.actor_lr)

        self.init_temp = algo_args.init_temp
        self.log_temp = torch.nn.Parameter(torch.tensor(math.log(self.init_temp),device=self.device,dtype=torch.float32))
        self.learn_temp = algo_args.learn_temp
        if self.learn_temp:
            self.temp_optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.temp_optimizer]([self.log_temp], lr=self.temp_lr)
        self.update_log_temp = algo_args.update_log_temp
        
        # target_entropy is expressed per action dimension (-1.0 is the usual SAC heuristic)
        self.target_entropy = algo_args.target_entropy * self.action_dim

    def _update_buffer(self, batch):
        self.buffer.add(batch)
    
    def random_choose_action(self):
        return self.interaction_step<self.start_train_step


    def _update_policy(self):
        return self._sac_update(self.buffer.sample(self.batch_size))

    def _sac_update(self, batch):
        """One SAC update on an already-sampled batch.

        ``absorbing`` / ``next_absorbing`` are optional and only set by DAC: they
        mark the artificial absorbing states, where the agent has no control and
        therefore neither earns an entropy bonus nor shapes the temperature.
        """
        self.agent.train()
        with Result("train") as result:
            states, actions, next_states, rewards, dones = batch['states'], batch['actions'], batch['next_states'], batch['rewards'], batch['dones']

            rewards = torch.from_numpy(rewards).float().to(self.device).unsqueeze(1)
            dones = torch.from_numpy(dones).float().to(self.device).unsqueeze(1)
            nstep_gamma = torch.from_numpy(batch['nstep_gamma']).float().to(self.device).unsqueeze(1)

            zeros = np.zeros(len(batch['states']), dtype=np.float32)
            absorbing = torch.from_numpy(batch.get('absorbing', zeros)).float().to(self.device).unsqueeze(1)
            next_absorbing = torch.from_numpy(batch.get('next_absorbing', zeros)).float().to(self.device).unsqueeze(1)

            # calculate the q loss    
            with torch.no_grad():
                target = rewards + (1 - dones) * nstep_gamma * self.agent.get_value(next_states,temperature=self.log_temp.detach().exp(),absorbing=next_absorbing)
                    
            q1, q2 = self.agent.get_double_q_function(states, actions)
            
            td_error1 = target - q1
            td_error2 = target - q2
            

            q_loss = torch.mean(td_error1**2) + torch.mean(td_error2**2)
            
            # do gradient update to the agent 
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()



            # update actor
            
            rsample_actions, log_probs, u = self.agent.resample_action(states)
            control = 1 - absorbing.squeeze(1)

            q = self.agent.get_q_function(states, rsample_actions).squeeze(1)
            actor_loss = (control * (self.log_temp.detach().exp() * log_probs - q)).sum()/control.sum()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.learn_temp:
                if self.update_log_temp:
                    # why here use the log_temp instead of temp?
                    temp_loss = -self.log_temp * (control * (log_probs.detach()+self.target_entropy)).sum()/control.sum()
                else:
                    temp_loss = self.log_temp.exp() * (control * (log_probs.detach()+self.target_entropy)).sum()/control.sum()
                self.temp_optimizer.zero_grad()
                temp_loss.backward()
                self.temp_optimizer.step()
            

            if self.use_target:
                if self.gradient_step%self.target_update_interval==0:
                    self.agent.update_target()


        with torch.no_grad():
            dist = self.agent.dist(states)
            mu, std = dist.base_dist.mean, dist.base_dist.stddev
            result.add_metric("actor/mean_action", mu.mean().item())
            result.add_metric("actor/std_action", std.mean().item())
        result.add_metric("critic/q_value",q.mean().item())
        result.add_metric("critic1/td_error_abs", torch.abs(td_error1).mean().item())
        result.add_metric("critic2/td_error_abs", torch.abs(td_error2).mean().item())
        result.add_metric("critic/q_loss", q_loss.item())
        result.add_metric("actor/loss", actor_loss.item())
        result.add_metric("actor/entropy", -log_probs.mean().item())
        if self.learn_temp:
            result.add_metric("temp/loss", temp_loss.item())
            result.add_metric("temp/value", self.log_temp.exp().item())
        
        self.gradient_step += 1
        return result
