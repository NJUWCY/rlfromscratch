import gymnasium as gym
import torch 
from torch.distributions import kl_divergence
import numpy as np
from typing import Tuple

from logger.logger import Logger
from memory.memory import ReplayBuffer, TrajectoryRollout
from agent.agent import AgentBase
from utils.result import Result
from utils.utils import get_flat_grad, conjugate_gradients, kl_product, get_model_flat_parameters, load_flat_parameters_to_model, OPTIMIZER_DICT
from .baseonpolicy import OnPolicyAlgorithm

class TRPO(OnPolicyAlgorithm):

    """Base class for off-policy RL algorithms."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: AgentBase, logger: Logger, device, save_pth: str, best_pth:str, args):
        super(TRPO,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth, best_pth, args)

        algo_args = args.algorithm

        self.delta = algo_args.delta
        self.cg_steps = algo_args.cg_steps
        self.linesearch_coeffient = algo_args.linesearch_coeffient
        self.backtracking_steps = algo_args.backtracking_steps
        self.lr = algo_args.learning_rate
        self.lambda_ = algo_args.lambda_
        self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](self.agent.critic.parameters(), lr=self.lr)
        self.advan_norm = algo_args.advan_norm
        self.critic_update_steps = algo_args.critic_update_steps
        self.critic_batch_size = algo_args.critic_batch_size
        self.eigenvalue_reg = algo_args.eigenvalue_reg
        

    def _update_buffer(self, batch):
        self.buffer.add(batch)
    
    def compute_advantages_from_traj(self)->Tuple[np.ndarray,np.ndarray]:
        states = self.traj_rollout.states
        rewards = self.traj_rollout.rewards 
        dones = self.traj_rollout.dones 
        

        with torch.no_grad():
            values = self.agent.get_value(states.reshape((-1,states.shape[-1]))).squeeze(1).reshape((self.trajnum, self.max_episode_length)).cpu().numpy()

        delta = np.zeros((self.trajnum,),dtype=np.float32)
        last_advan = np.zeros((self.trajnum,),dtype=np.float32)
        last_value = np.zeros((self.trajnum,),dtype=np.float32)
        last_returns = np.zeros((self.trajnum,),dtype=np.float32)
        advantages = np.zeros((self.trajnum,self.max_episode_length),dtype=np.float32)
        returns = np.zeros((self.trajnum,self.max_episode_length),dtype=np.float32)

        for i in range(self.max_episode_length-1,-1,-1):
            delta = rewards[:,i] + self.gamma*(1-dones[:,i])*last_value - values[:,i]
            advantages[:,i] = delta + self.gamma*self.lambda_*(1-dones[:,i])*last_advan
            returns[:,i] = rewards[:,i] + (1-dones[:,i])*self.gamma*last_returns
            
            
            last_value = values[:,i]
            last_advan = advantages[:,i]
            last_returns = returns[:,i]
        return returns, advantages

    def compute_advantages_from_rollout(self)->Tuple[np.ndarray,np.ndarray]:
        states = self.buffer.buffer['states']
        rewards = self.buffer.buffer['rewards']
        dones = self.buffer.buffer['dones']
        
        num_envs, rollout_length = states.shape[0], states.shape[1]

        with torch.no_grad():
            values = self.agent.get_value(states.reshape((-1,states.shape[-1]))).squeeze(1).reshape((num_envs, rollout_length)).cpu().numpy()

        delta = np.zeros((num_envs,),dtype=np.float32)
        last_advan = np.zeros((num_envs,),dtype=np.float32)
        with torch.no_grad():
            last_value = self.agent.get_value(self.observations).squeeze(1).cpu().numpy()
        advantages = np.zeros((num_envs,rollout_length),dtype=np.float32)

        for i in range(rollout_length-1,-1,-1):
            delta = rewards[:,i] + self.gamma*(1-dones[:,i])*last_value - values[:,i]
            advantages[:,i] = delta + self.gamma*self.lambda_*(1-dones[:,i])*last_advan
            
            
            last_value = values[:,i]
            last_advan = advantages[:,i]
        returns = advantages + values
        return returns, advantages

        
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
                advantages = (advantages-advantages.mean())/advantages.std()



            log_prob = self.agent.log_prob(states,actions)

            ratios = (log_prob - old_log_probs).exp()

            surrogate_target = torch.mean(ratios*advantages) 

            surrogate_gradient = get_flat_grad(surrogate_target, self.agent.actor).detach() # retain_graph=True

            dist = self.agent.dist(states)
            with torch.no_grad():
                old_dist = self.agent.dist(states)
            
            kl = kl_divergence(old_dist, dist).mean()

            kl_grad = get_flat_grad(kl, self.agent.actor, create_graph=True)

            gradient_direction = conjugate_gradients(self.agent.actor,surrogate_gradient, kl_grad, self.cg_steps,self.eigenvalue_reg)
            
            stepsize = torch.sqrt(
                2*self.delta / (   torch.sum(gradient_direction*kl_product(gradient_direction,kl_grad,self.agent.actor, self.eigenvalue_reg))   )
                                )


            # linesearch

            with torch.no_grad():
                backtrack_step = 0
                flat_params = get_model_flat_parameters(self.agent.actor)
                for i in range(self.backtracking_steps):
                    new_parameter = flat_params + stepsize*gradient_direction

                    load_flat_parameters_to_model(new_parameter, self.agent.actor)
                    new_dist = self.agent.dist(states)
                    new_log_prob = new_dist.log_prob(actions)
                    new_ratios = (new_log_prob-old_log_probs).exp()
                    new_surrogate_target = torch.mean(new_ratios*advantages)

                    new_kl = kl_divergence(old_dist, new_dist).mean()
                    
                    if new_surrogate_target>surrogate_target and new_kl < self.delta:
                        backtrack_step = i
                        break
                    
                    stepsize *= self.linesearch_coeffient
                    if i==self.backtracking_steps-1:
                        backtrack_step = i
                        load_flat_parameters_to_model(flat_params,self.agent.actor)
            
            # update the value function 
            dataset_size = states.size(0)
            batch_size = min(self.critic_batch_size, dataset_size)

            for _ in range(self.critic_update_steps):
                indices = torch.randperm(dataset_size, device=states.device)

                for start in range(0, dataset_size, batch_size):
                    batch_idx = indices[start:start + batch_size]
                    batch_states = states[batch_idx]
                    batch_returns = returns[batch_idx]

                    values = self.agent.get_value(batch_states).squeeze(1)
                    value_loss = ((values - batch_returns) ** 2).mean()

                    self.optimizer.zero_grad()
                    value_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 1.0)
                    self.optimizer.step()


        result.add_metric("value/loss", value_loss.item())
        result.add_metric("actor/backtrack_step",backtrack_step)
        result.add_metric("actor/new_surrogate_target",new_surrogate_target.item())
        result.add_metric("actor/new_kl",new_kl.item())
        if not self.advan_norm:
            result.add_metric("actor/adv_mean", advantages.mean().item())
            result.add_metric("actor/adv_max", torch.max(advantages).item())
        result.add_metric("actor/xHx",torch.sum(gradient_direction*kl_product(gradient_direction,kl_grad,self.agent.actor,self.eigenvalue_reg)).item())
        result.add_metric("actor/stepsize",stepsize.item())
        result.add_metric("actor/gradient_direction_l2norm",torch.norm(gradient_direction,p=2).item())
        result.add_metric("actor/new_param_l2norm",torch.norm(new_parameter).item())
        return result


                
