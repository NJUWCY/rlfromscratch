
import numpy as np
import random
import gymnasium as gym 
from utils.utils import get_action_dim
from .datastructure import SumTree



class ReplayBuffer:
    """Replay buffer used to store and sample transitions when training an off-policy model.
    Here we use the implementation of memory efficient replay buffer."""
    def __init__(self,observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Space,
                 buffer_size: int,
                 num_envs: int, 
                 onpolicy=False):
        
        self.buffer_size = buffer_size
        action_dim = get_action_dim(action_space)
        self.num_envs = num_envs
        self.buffer_message = {
            "states": {"shape":(num_envs, buffer_size, *observation_space.shape), "dtype":observation_space.dtype} ,
            "actions": {"shape": (self.num_envs, buffer_size, action_dim), "dtype": action_space.dtype},
            "dones": {"shape": (self.num_envs, buffer_size), "dtype": np.float32},
            "rewards": {"shape": (self.num_envs, buffer_size), "dtype": np.float32},
            "next_states": {"shape": (self.num_envs, buffer_size, *observation_space.shape), "dtype": observation_space.dtype}
        }
        self.onpolicy = onpolicy
        if onpolicy:
            self.buffer_message['log_probs'] = {"shape": (self.num_envs, buffer_size), "dtype": np.float32}
            self.buffer_message['truncateds'] = {"shape": (self.num_envs, buffer_size), "dtype": np.float32}
        
        self.buffer = {}
    
        # we think the replay buffer as a big circle 
        self.pos = 0
        self.full = False 
        self.reset()


    def add(self,batch: dict[str, np.ndarray]):
        """Add a batch of transitions to the replay buffer."""
        # for onpolicy algorithm, we should not allow the cover of the old data
        if self.onpolicy:
            assert not self.full

        s = batch['states']

        num_envs,batch_size = s.shape[0],s.shape[1]   # the first dimension is the batch size, the second dimension is the number of envs
    
        end = self.pos + batch_size


        indices = (self.pos + np.arange(batch_size))%self.buffer_size

        if end <= self.buffer_size:
            # not crossing the boundary
            for key in self.buffer:
                self.buffer[key][:,self.pos:end] = batch[key]
        else:
            # crossing the boundary, we need to split the batch into two parts
            first_part = self.buffer_size - self.pos
            second_part = batch_size - first_part
            for key in self.buffer:
                self.buffer[key][:,self.pos:] = batch[key][:,:first_part]
                self.buffer[key][:,:second_part] = batch[key][:,first_part:]

        if end>=self.buffer_size:
            self.full = True
        self.pos = end % self.buffer_size

        return indices 
        
       

    def _sample_from_indices(self, batch_indices: np.ndarray) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer.
        TODO: add more states preprocess to other environments 
        """
        env_indices = np.random.randint(0, high=self.num_envs, size=(len(batch_indices),))
        # this sample method is based one the dot product of env_indices and batch_indices 
        batch = {}
        for key in self.buffer:
            batch[key] = self.buffer[key][env_indices, batch_indices]
        

        return batch

    def _get_indices(self, batch_size: int) -> np.ndarray:
        """Get a batch of indices to sample from the replay buffer.
        You may change this method to use prioritized experience replay or other sampling strategies."""
        if self.full:
            batch_indices = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            batch_indices = np.random.randint(0, self.pos, size=batch_size)
        
        return batch_indices

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer."""
        # This is only used for off-policy
        
        batch_indices = self._get_indices(batch_size)
        return self._sample_from_indices(batch_indices)


    def reset(self):
        """ clear the replay buffer."""
        for key, message in self.buffer_message.items():
            if key != "dones":
                self.buffer[key] = np.zeros(shape=message['shape'],dtype=message['dtype'])
            else:
                self.buffer[key] = np.ones(shape=message['shape'],dtype=message['dtype'])

        self.pos = 0
        self.full = False 


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    This implementation is based on the proportional prioritized replay buffer.
    """
    def __init__(self,observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Space,
                 buffer_size: int,
                 num_envs: int,
                 alpha: float,
                 beta: float,
                 epsilon=1e-6,
                 batch_norm=False,
                 onpolicy=False):

        assert num_envs==1, "We only implement Prioritized Replaybuffer for num_envs==1"
        super(PrioritizedReplayBuffer, self).__init__(observation_space, action_space, buffer_size, num_envs, onpolicy)

        self.alpha = alpha 
        self.beta = beta 
        self.epsilon = epsilon
        self.priority = SumTree(buffer_size)

        self.max_td = 1.

        self.batch_norm = batch_norm

    def add(self,batch: dict[str, np.ndarray]):
        indices = super().add(batch)
        initial_priority = np.ones_like(indices,dtype=np.float64)*(self.max_td ** self.alpha)
        self.priority[indices] = initial_priority
        return indices

    
    def _get_indices(self,batch_size:int):
        # Note: Here we use the stratified sampling according to the implementation in RainBow
        
        p_sum = self.priority.tree[1]
        segment = p_sum / batch_size
        ratios = np.arange(batch_size,dtype=np.float64) * segment 
        ratios += np.random.uniform(0.0, segment, [batch_size])
        
        indices = self.priority.get_prefix_idx(ratios)
        return indices

    def _get_weight(self, indices:np.ndarray):
        # Note: We do the normalization inside the batch
        weights = (self.priority[indices])**(-self.beta)
        weights = weights / weights.max()
        return weights

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer."""
        # This is only used for off-policy
        
        batch_indices = self._get_indices(batch_size)
        batch = self._sample_from_indices(batch_indices)
        weights = self._get_weight(batch_indices)
        batch['weights'] = weights 
        batch['indices'] = batch_indices
        return batch 


    def update_priority(self, indices:np.ndarray, td_error:np.ndarray):
        assert indices.shape==td_error.shape
        td_abs = np.abs(td_error) + self.epsilon
        priorities = td_abs**self.alpha
        self.max_td = max(self.max_td, td_abs.max())
        self.priority[indices] = priorities
        


    def reset(self):
        super().reset()
        self.priority = SumTree(self.buffer_size)
        self.max_td = 1.
    

    

        


class TrajectoryRollout:
    def __init__(self, observation_space: gym.spaces.Box, 
                 action_space: gym.spaces.Space,
                 trajnum: int,
                 max_episode_length: int):
        self.trajnum = trajnum 
        self.max_episode_length = max_episode_length
        self.action_dim = get_action_dim(action_space)

        self.observation_space = observation_space 
        self.action_space = action_space
        # TODO: use dict to manage these values to avoid miss them in the reset
        self.states = np.zeros((trajnum, max_episode_length, *observation_space.shape), dtype=observation_space.dtype) 
        self.actions = np.zeros((trajnum, max_episode_length, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((trajnum, max_episode_length), dtype=np.float32)
        self.dones = np.ones((trajnum, max_episode_length), dtype=np.float32) # This is specifically designed to calculate the rewards to go 
        self.masks = np.zeros((trajnum, max_episode_length),dtype=np.float32)
        self.log_probs = np.zeros((trajnum,max_episode_length),dtype=np.float32)
        self.last_states = np.zeros((trajnum, *observation_space.shape), dtype=observation_space.dtype) # store the last states of each trajectory for calculating the value of the last states in the GAE calculation
        self.truncateds = np.zeros((trajnum, ), dtype=np.float32) # store the truncateds of each step for calculating the returns in the case of truncation
        self.pos = 0
        self.full = False 
    
    def add_traj(self, traj_batch: dict):
        if self.full:
            raise ValueError("Trajectory Rollout is full, you need to clear the rollout first.")
        s, a, r, d, lp, ls, tr = traj_batch['states'], traj_batch['actions'], traj_batch['rewards'], traj_batch['dones'], traj_batch['log_probs'], traj_batch['last_states'], traj_batch['truncateds']
        traj_length = len(s)
    
        
        self.states[self.pos,-traj_length:] = s
        self.actions[self.pos,-traj_length:] = a
        self.rewards[self.pos,-traj_length:] = r 
        self.dones[self.pos, -traj_length:] = d
        self.log_probs[self.pos, -traj_length:] = lp
        self.masks[self.pos,-traj_length:] = 1
        self.last_states[self.pos] = ls
        self.truncateds[self.pos] = tr
        self.pos += 1
        if self.pos>=self.trajnum:
            self.full = True

    
    def reset(self):
        """ clear the replay buffer."""
        trajnum = self.trajnum 
        max_episode_length = self.max_episode_length
        observation_space = self.observation_space
        action_space = self.action_space

        self.states = np.zeros((trajnum, max_episode_length, *observation_space.shape), dtype=observation_space.dtype) 
        self.actions = np.zeros((trajnum, max_episode_length, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((trajnum, max_episode_length), dtype=np.float32)
        self.dones = np.ones((trajnum, max_episode_length), dtype=np.float32) # This is specifically designed to calculate the rewards to go 
        self.masks = np.zeros((trajnum, max_episode_length),dtype=np.float32)
        self.log_probs = np.zeros((trajnum,max_episode_length),dtype=np.float32)
        self.last_states = np.zeros((trajnum, *observation_space.shape), dtype=observation_space.dtype) # store the last states of each trajectory for calculating the value of the last states in the GAE calculation
        self.truncateds = np.zeros((trajnum, ), dtype=np.float32) # store the truncateds of each step for calculating the returns in the case of truncation
        self.pos = 0
        self.full = False 
    




if __name__ == "__main__":
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    buffer_size = 8
    num_envs = 1
    alpha = 0.6
    beta = 0.4

    buf = PrioritizedReplayBuffer(obs_space, act_space, buffer_size, num_envs, alpha, beta)

    # 手动构造两批数据，每批 4 个 transition
    batch1 = {
        "states": np.array([[[1, 0, 0, 0],
                             [2, 0, 0, 0],
                             [3, 0, 0, 0],
                             [4, 0, 0, 0]]], dtype=np.float32),
        "actions": np.array([[[0], [1], [0], [1]]], dtype=np.int64),
        "rewards": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        "dones": np.array([[0, 0, 0, 1]], dtype=np.float32),
        "next_states": np.array([[[2, 0, 0, 0],
                                  [3, 0, 0, 0],
                                  [4, 0, 0, 0],
                                  [5, 0, 0, 0]]], dtype=np.float32),
    }

    batch2 = {
        "states": np.array([[[5, 0, 0, 0],
                             [6, 0, 0, 0],
                             [7, 0, 0, 0],
                             [8, 0, 0, 0]]], dtype=np.float32),
        "actions": np.array([[[1], [0], [1], [0]]], dtype=np.int64),
        "rewards": np.array([[5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
        "dones": np.array([[0, 0, 1, 0]], dtype=np.float32),
        "next_states": np.array([[[6, 0, 0, 0],
                                  [7, 0, 0, 0],
                                  [8, 0, 0, 0],
                                  [9, 0, 0, 0]]], dtype=np.float32),
    }

    print("=== 添加 batch1 (indices 0-3) ===")
    indices1 = buf.add(batch1)
    print(f"indices: {indices1}")
    print(f"SumTree leaves: {buf.priority.tree[buf.priority.bound:][:buffer_size]}")

    print("\n=== 添加 batch2 (indices 4-7) ===")
    indices2 = buf.add(batch2)
    print(f"indices: {indices2}")
    print(f"SumTree leaves: {buf.priority.tree[buf.priority.bound:][:buffer_size]}")
    print(f"SumTree root (total priority): {buf.priority.tree[1]}")

    # 手动指定 td_error 来更新优先级
    td_errors = np.array([0.1, 0.5, 2.0, 0.3, 1.0, 0.2, 3.0, 0.8])
    all_indices = np.arange(buffer_size)
    print(f"\n=== 更新优先级, td_errors = {td_errors} ===")
    buf.update_priority(all_indices, td_errors)
    print(f"SumTree leaves: {buf.priority.tree[buf.priority.bound:][:buffer_size]}")
    print(f"SumTree root (total priority): {buf.priority.tree[1]}")
    print(f"max_td: {buf.max_td}, min_td: {buf.min_td}")

    # 采样并查看 weights
    print("\n=== 采样 batch_size=4 ===")
    batch = buf.sample(4)
    print(f"sampled states[:,0]: {batch['states'][:, 0]}")
    print(f"sampled rewards: {batch['rewards']}")
    print(f"weights: {batch['weights']}")

