from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import gymnasium as gym
import torch 
from typing import Tuple


from logger.logger import Logger
from memory.memory import ReplayBuffer, TrajectoryRollout
from agent.agent import AgentBase, AtariDQNAgent
from utils.result import Result
from utils.utils import to_useful_action
from .basealgorithm import BaseAlgorithm


class OnPolicyAlgorithm(BaseAlgorithm, ABC):

    """Base class for off-policy RL algorithms."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: AgentBase, logger: Logger, device, save_pth: str, best_pth:str, args):
        super(OnPolicyAlgorithm,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth,best_pth, args)

        self.collect_traj = args.algorithm.collect_traj
        if self.collect_traj:
            self.traj_rollout = buffer
            self.trajnum = buffer.trajnum 
            self.max_episode_length = buffer.max_episode_length
        self.gae = args.algorithm.gae
        self.lambda_ = args.algorithm.lambda_
           
            
    
    def update(self, batch, start_train)-> Result:
        # For on-policy training, it is defaulted to do the training 
        if not self.collect_traj: # If collect the trajector, the update of the buffer is done in the collect_trajectories function
            with Result("buffer") as result:
                self._update_buffer(batch)
            update_policy_log = self._update_policy()
            result.add(update_policy_log)
            self.buffer.reset()
        else:
            result = self._update_policy()
            self.traj_rollout.reset() # for on-policy training, we will reset the replay buffer after each update
        return result
        
        
    
    @abstractmethod
    def _update_buffer(self, batch):
        """
        update the replay buffer with given selected batch of data
        :param batch: the batch of data
        """
    
    @abstractmethod
    def _update_policy(self):
        """
        do gradient update to the policy with a batch of data sampled from replay buffer
        """

    def collect_trajectories(self):
        traj_states = [[] for _ in range(self.num_training_envs)]
        traj_actions = [[] for _ in range(self.num_training_envs)]
        traj_rewards = [[] for _ in range(self.num_training_envs)]
        traj_dones = [[] for _ in range(self.num_training_envs)]
        traj_log_probs = [[] for _ in range(self.num_training_envs)]
        # traj_truncateds = [[] for _ in range(self.num_training_envs)]
        with Result("interact") as result:
            while not self.traj_rollout.full:
                with torch.no_grad():
                    actions, action_infos = self.agent.select_action(self.observations, self.train_action_deterministic)
                    log_probs = action_infos['log_probs']
                    
                    next_observations, rewards, terminateds, infos = self.training_envs.step(to_useful_action(self.action_space, self.action_dim, actions))

                    traj_states = [traj_states[i]+[self.observations[i]] for i in range(self.num_training_envs)]
                    traj_actions = [traj_actions[i]+[actions[i]] for i in range(self.num_training_envs)]
                    traj_rewards = [traj_rewards[i]+[rewards[i]] for i in range(self.num_training_envs)]
                    traj_dones = [traj_dones[i]+[terminateds[i]] for i in range(self.num_training_envs)]
                    traj_log_probs = [traj_log_probs[i]+[log_probs[i]] for i in range(self.num_training_envs)]
                    # traj_truncateds = [traj_truncateds[i]+[infos[i].get("TimeLimit.truncated",False)] for i in range(self.num_training_envs)]

                    for i in range(self.num_training_envs):
                        if self.traj_rollout.full:
                            break
                        info = infos[i]
                        if "episode" in info:
                            self.episode_reward_buffer.append(info['episode']['r'])
                            self.traj_rollout.add_traj(
                                dict(
                                    states=np.array(traj_states[i]),
                                    actions=np.array(traj_actions[i]),
                                    rewards=np.array(traj_rewards[i]),
                                    dones=np.array(traj_dones[i]),
                                    log_probs=np.array(traj_log_probs[i])
                                )
                            )
                            traj_states[i] = []
                            traj_actions[i] = []
                            traj_rewards[i] = []
                            traj_dones[i] = []
                            traj_log_probs[i] = []
                            
                            
                    self.observations = next_observations
                    self.interaction_step += self.num_training_envs 
        # reset the environment after each collection
        self.initialize()
        if len(self.episode_reward_buffer)>0:
            result.add_metric("reward_mean",np.mean(self.episode_reward_buffer))
            result.add_metric("reward_std",np.std(self.episode_reward_buffer))
        return None, result 



    def start_train(self):
        return True
    
    def interact_with_envs(self):
        if self.collect_traj:
            return self.collect_trajectories()
        else:
            return super().interact_with_envs()
    
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
        if self.gae:
            return returns, advantages
        else:
            return returns, returns - values

    def compute_advantages_from_rollout(self)->Tuple[np.ndarray,np.ndarray]:
        states = self.buffer.buffer['states']
        rewards = self.buffer.buffer['rewards']
        dones = self.buffer.buffer['dones']
        
        num_envs, rollout_length = states.shape[0], states.shape[1]

        with torch.no_grad():
            values = self.agent.get_value(states.reshape((-1,states.shape[-1]))).squeeze(1).reshape((num_envs, rollout_length)).cpu().numpy()
            last_value = self.agent.get_value(self.observations).squeeze(1).cpu().numpy()

        advantages = np.zeros((num_envs,rollout_length),dtype=np.float32)
        returns = np.zeros((num_envs,rollout_length),dtype=np.float32)
        if self.gae:
            delta = np.zeros((num_envs,),dtype=np.float32)
            last_advan = np.zeros((num_envs,),dtype=np.float32)
    
            for i in range(rollout_length-1,-1,-1):
                delta = rewards[:,i] + self.gamma*(1-dones[:,i])*last_value - values[:,i]
                advantages[:,i] = delta + self.gamma*self.lambda_*(1-dones[:,i])*last_advan
                
                
                last_value = values[:,i]
                last_advan = advantages[:,i]
            returns = advantages + values
            return returns, advantages
        else:
            last_returns = last_value
            for i in range(rollout_length-1,-1,-1):
                returns[:,i] = rewards[:,i] + (1-dones[:,i])*self.gamma*last_returns
                last_returns = returns[:,i]
            return returns, returns - values






