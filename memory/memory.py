
from pathlib import Path

import h5py
import numpy as np
import random
import gymnasium as gym 
from utils.utils import get_action_dim
from env.absorbing import absorbing_state, absorbing_action, insert_absorbing_self_loops
from .datastructure import SumTree



class ReplayBuffer:
    """Replay buffer used to store and sample transitions when training an off-policy model.
    Here we use the implementation of memory efficient replay buffer."""
    def __init__(self,observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Space,
                 buffer_size: int,
                 num_envs: int, 
                 gamma:float=0.99,
                 nstep:int=1,
                 onpolicy=False,
                 store_u=False
                 ):
        
        self.buffer_size = buffer_size
        action_dim = get_action_dim(action_space)
        self.num_envs = num_envs
        # Note: Here we use the float32 to store the observations to avoid the precision issue and OpenMP parallel issue.
        obs_dtype = np.float32 if observation_space.dtype == np.float64 else observation_space.dtype
        self.buffer_message = {
            "states": {"shape":(num_envs, buffer_size, *observation_space.shape), "dtype":obs_dtype},
            "actions": {"shape": (self.num_envs, buffer_size, action_dim), "dtype": action_space.dtype},
            "dones": {"shape": (self.num_envs, buffer_size), "dtype": np.float32},
            "rewards": {"shape": (self.num_envs, buffer_size), "dtype": np.float32},
            "next_states": {"shape": (self.num_envs, buffer_size, *observation_space.shape), "dtype": obs_dtype}
        }
        self.buffer_message['truncateds'] = {"shape": (self.num_envs, buffer_size), "dtype": np.float32}
        self.onpolicy = onpolicy
        if onpolicy:
            self.buffer_message['log_probs'] = {"shape": (self.num_envs, buffer_size), "dtype": np.float32}
            if store_u:
                self.buffer_message['u'] = {"shape": (self.num_envs, buffer_size, action_dim), "dtype": np.float32}
        
        self.buffer = {}
    
        # we think the replay buffer as a big circle 
        self.pos = 0
        self.full = False 

        self.nstep = nstep
        self.gamma = gamma

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

    def _is_outof_range(self, indices:np.ndarray):
        if self.full:
            return indices>=self.buffer_size 
        else:
            return indices>=self.pos
    
    def _loop_to_real(self, loop_indices:np.ndarray):
        if self.full:
            return (loop_indices+self.pos)%self.buffer_size 
        else:
            return loop_indices 
    
    def _compute_nstep(self,env_indices:np.ndarray, indices: np.ndarray, batch: dict):
        if self.full:
            loop_indices = indices - self.pos 
            loop_indices[loop_indices<0] += self.buffer_size 
        else:
            loop_indices = np.copy(indices)


        end_loop_indices = np.copy(loop_indices)
        returns = np.copy(batch['rewards'])
        
        for i in range(1,self.nstep):

            outof_range = self._is_outof_range(end_loop_indices+1)

            end_indices = self._loop_to_real(end_loop_indices)
            done = np.logical_or(self.buffer['dones'][env_indices, end_indices], \
                           self.buffer['truncateds'][env_indices, end_indices])
            
            end_loop_indices += 1
            end_loop_indices[np.logical_or(outof_range, done)] -= 1

            cur_indices = loop_indices + i
            outof_end = cur_indices > end_loop_indices
            cur_indices[outof_end] = 0 # random choose an index 
            returns += self.gamma**i * (1-outof_end) * self.buffer['rewards'][env_indices, self._loop_to_real(cur_indices)]

        end_real_indices = self._loop_to_real(end_loop_indices)
        batch['rewards'] = returns
        batch['dones'] = self.buffer['dones'][env_indices, end_real_indices]
        batch['truncateds'] = self.buffer['truncateds'][env_indices, end_real_indices]
        batch['next_states'] = self.buffer['next_states'][env_indices, end_real_indices]
        batch['nstep_gamma'] = self.gamma ** (end_loop_indices - loop_indices + 1)
        return batch 
       

    def _sample_from_indices(self, batch_indices: np.ndarray) -> dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer.
        TODO: add more states preprocess to other environments 
        """
        env_indices = np.random.randint(0, high=self.num_envs, size=(len(batch_indices),))
        # this sample method is based one the dot product of env_indices and batch_indices 
        batch = {}
        for key in self.buffer:
            batch[key] = self.buffer[key][env_indices, batch_indices]
        
        batch = self._compute_nstep(env_indices, batch_indices, batch)
        

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
                 gamma=0.99,
                 nstep=1,
                 onpolicy=False):

        assert num_envs==1, "We only implement Prioritized Replaybuffer for num_envs==1"
        super(PrioritizedReplayBuffer, self).__init__(observation_space, action_space, buffer_size, num_envs, gamma, nstep, onpolicy)

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
    

class AbsorbingReplayBuffer(ReplayBuffer):
    """Replay buffer for DAC, one circular stream per environment.

    Absorbing self-loops are inserted immediately after real terminations, so
    the number of writes per step differs across environments. Each environment
    therefore keeps its own write head. n-step returns walk that environment's
    timeline and stop on ``dones``, ``truncateds``, or an absorbing self-loop,
    so they never cross an episode boundary or another environment.
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Space,
                 buffer_size: int,
                 gamma: float = 0.99,
                 nstep: int = 1,
                 num_envs: int = 1):
        super(AbsorbingReplayBuffer, self).__init__(
            observation_space, action_space, buffer_size, num_envs=num_envs,
            gamma=gamma, nstep=nstep, onpolicy=False,
        )
        for key in ("absorbing", "next_absorbing"):
            self.buffer_message[key] = {"shape": (num_envs, buffer_size), "dtype": np.float32}
        self.reset()

    def reset(self):
        super().reset()
        self.env_pos = np.zeros(self.num_envs, dtype=np.int64)
        self.env_full = np.zeros(self.num_envs, dtype=bool)
        self.pos = 0
        self.full = False

    def filled(self) -> int:
        return int(np.where(self.env_full, self.buffer_size, self.env_pos).sum())

    def add(self, batch: dict[str, np.ndarray]):
        raise NotImplementedError("AbsorbingReplayBuffer stores per-env transitions, use add_transitions")

    def add_transitions(self, batch: dict[str, np.ndarray], env_index: int = 0):
        """Append a time-major block of transitions to one environment's stream."""
        n = len(batch['states'])
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if n > self.buffer_size:
            raise ValueError(f"Cannot add {n} transitions to a buffer of size {self.buffer_size}")

        pos = int(self.env_pos[env_index])
        indices = (pos + np.arange(n)) % self.buffer_size
        for key in self.buffer:
            self.buffer[key][env_index, indices] = batch[key]

        end = pos + n
        if end >= self.buffer_size:
            self.env_full[env_index] = True
        self.env_pos[env_index] = end % self.buffer_size
        self.pos = int(self.env_pos[0])
        self.full = bool(self.env_full[0]) if self.num_envs == 1 else bool(np.all(self.env_full))
        return indices

    def _env_sizes(self) -> np.ndarray:
        return np.where(self.env_full, self.buffer_size, self.env_pos)

    def _get_indices(self, batch_size: int):
        sizes = self._env_sizes().astype(np.float64)
        total = sizes.sum()
        if total <= 0:
            raise ValueError("Cannot sample from an empty AbsorbingReplayBuffer")
        env_indices = np.random.choice(self.num_envs, size=batch_size, p=sizes / total)
        batch_indices = (np.random.random(batch_size) * sizes[env_indices]).astype(np.int64)
        return env_indices, batch_indices

    def _loop_indices(self, env_indices: np.ndarray, indices: np.ndarray):
        pos = self.env_pos[env_indices]
        full = self.env_full[env_indices]
        loop = np.where(full, (indices - pos) % self.buffer_size, indices)
        return loop, pos, full

    def _real_indices(self, loop_indices: np.ndarray, pos: np.ndarray, full: np.ndarray) -> np.ndarray:
        return np.where(full, (loop_indices + pos) % self.buffer_size, loop_indices)

    def _compute_nstep(self, env_indices: np.ndarray, indices: np.ndarray, batch: dict):
        loop_indices, pos, full = self._loop_indices(env_indices, indices)
        end_loop_indices = np.copy(loop_indices)
        returns = np.copy(batch['rewards'])
        limit = np.where(full, self.buffer_size, pos)

        batch_size = len(indices)
        nstep_states = np.zeros((batch_size, self.nstep, *self.buffer['states'].shape[2:]),
                                dtype=self.buffer['states'].dtype)
        nstep_actions = np.zeros((batch_size, self.nstep, *self.buffer['actions'].shape[2:]),
                                 dtype=self.buffer['actions'].dtype)
        nstep_mask = np.zeros((batch_size, self.nstep), dtype=np.float32)
        nstep_states[:, 0] = batch['states']
        nstep_actions[:, 0] = batch['actions']
        nstep_mask[:, 0] = 1.0

        for i in range(1, self.nstep):
            end_real = self._real_indices(end_loop_indices, pos, full)
            outof_range = (end_loop_indices + 1) >= limit
            stop = (
                (self.buffer['dones'][env_indices, end_real] > 0)
                | (self.buffer['truncateds'][env_indices, end_real] > 0)
                | (self.buffer['absorbing'][env_indices, end_real] > 0)
            )

            end_loop_indices = end_loop_indices + 1
            end_loop_indices[np.logical_or(outof_range, stop)] -= 1

            cur_loop = loop_indices + i
            in_window = cur_loop <= end_loop_indices
            cur_safe = np.where(in_window, cur_loop, 0)
            cur_real = self._real_indices(cur_safe, pos, full)
            returns += (self.gamma ** i) * in_window * self.buffer['rewards'][env_indices, cur_real]

            nstep_mask[:, i] = in_window.astype(np.float32)
            nstep_states[:, i] = self.buffer['states'][env_indices, cur_real]
            nstep_actions[:, i] = self.buffer['actions'][env_indices, cur_real]

        end_real_indices = self._real_indices(end_loop_indices, pos, full)
        batch['rewards'] = returns
        batch['dones'] = self.buffer['dones'][env_indices, end_real_indices]
        batch['truncateds'] = self.buffer['truncateds'][env_indices, end_real_indices]
        batch['next_states'] = self.buffer['next_states'][env_indices, end_real_indices]
        batch['next_absorbing'] = self.buffer['next_absorbing'][env_indices, end_real_indices]
        batch['nstep_gamma'] = self.gamma ** (end_loop_indices - loop_indices + 1)
        batch['nstep_states'] = nstep_states
        batch['nstep_actions'] = nstep_actions
        batch['nstep_mask'] = nstep_mask
        return batch

    def _sample_from_indices(self, env_indices: np.ndarray, batch_indices: np.ndarray) -> dict[str, np.ndarray]:
        batch = {}
        for key in self.buffer:
            batch[key] = self.buffer[key][env_indices, batch_indices]
        return self._compute_nstep(env_indices, batch_indices, batch)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        env_indices, batch_indices = self._get_indices(batch_size)
        return self._sample_from_indices(env_indices, batch_indices)


class TrajectoryRollout:
    def __init__(self, observation_space: gym.spaces.Box, 
                 action_space: gym.spaces.Space,
                 trajnum: int,
                 max_episode_length: int,
                 store_u=False):
        self.trajnum = trajnum 
        self.max_episode_length = max_episode_length
        self.action_dim = get_action_dim(action_space)
        self.store_u = store_u

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
        if self.store_u:
            self.u = np.zeros((trajnum, max_episode_length, self.action_dim), dtype=np.float32)
    
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
        if self.store_u:
            self.u[self.pos,-traj_length:] = traj_batch['u']
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


class ExpertDataset(ReplayBuffer):
    def __init__(self,file_path: str,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Space,
                 trajectory_num: int, 
                 subsample_frequency:int, 
                 gamma:float=0.99,
                 nstep:int=1,
                 absorbing:bool=False,
                 ):
        path = Path(file_path)
        suffix = path.suffix.lower()
        data: dict[str, np.ndarray] = {}
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as npz:
                data = {key: np.asarray(npz[key]) for key in npz.files}
        elif suffix in {".h5", ".hdf5"}:
            with h5py.File(path, "r") as f:
                def _collect(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        data[name] = np.asarray(obj[()])
                f.visititems(_collect)
        else:
            raise ValueError(
                f"Unsupported expert dataset format: {suffix}. "
                "Expected .npz, .h5, or .hdf5."
            )
        if not data:
            raise ValueError(f"Expert dataset is empty: {path}")

        self.subsample_frequency = subsample_frequency
        self.trajectory_num = trajectory_num

        for k, v in data.items():
            print(f"loaded expert dataset {k} with shape {v.shape}")

        if subsample_frequency <= 0:
            raise ValueError(f"subsample_frequency must be positive, got {subsample_frequency}")
        if trajectory_num <= 0:
            raise ValueError(f"trajectory_num must be positive, got {trajectory_num}")

        all_trajectories = self._split_trajectories(data)
        if len(all_trajectories) == 0:
            raise ValueError(f"No trajectories found in expert dataset: {path}")
        if trajectory_num > len(all_trajectories):
            raise ValueError(
                f"Requested trajectory_num={trajectory_num}, but only "
                f"{len(all_trajectories)} trajectories are available."
            )

        selected_idx = np.random.choice(
            len(all_trajectories), size=trajectory_num, replace=False
        )
        # list of trajectory_num dicts; each dict holds subsampled transitions
        trajectories: list[dict[str, np.ndarray]] = [
            {key: value[::subsample_frequency] for key, value in all_trajectories[i].items()}
            for i in selected_idx
        ]

        for i, (traj_idx, traj) in enumerate(zip(selected_idx, trajectories)):
            if "rewards" not in traj:
                raise KeyError("Expert trajectories must contain rewards to compute returns.")
            rewards = np.asarray(traj["rewards"], dtype=np.float64).reshape(-1)
            # Undiscounted episode return; also report discounted return with gamma.
            undiscounted_return = float(rewards.sum())
            discounts = gamma ** np.arange(rewards.shape[0], dtype=np.float64)
            discounted_return = float((discounts * rewards).sum())
            print(
                f"sampled expert trajectory {i}: original_idx={int(traj_idx)}, "
                f"length={rewards.shape[0]}, "
                f"return={undiscounted_return:.4f}, "
                f"discounted_return(gamma={gamma})={discounted_return:.4f}"
            )

        lengths = [len(traj['actions']) for traj in trajectories]
        total_length = sum(lengths)

        # Merge subsampled trajectories into the flat ReplayBuffer layout:
        # buffer[key].shape == (num_envs, buffer_size, ...)
        merged = {
            key: np.concatenate([traj[key] for traj in trajectories], axis=0)
            for key in trajectories[0]
        }

        def _pick(*candidates: str) -> np.ndarray | None:
            for name in candidates:
                if name in merged:
                    return merged[name]
            return None

        observations = _pick("observations", "states")
        next_observations = _pick("next_observations", "next_states")
        actions = _pick("actions")
        rewards = _pick("rewards")
        terminals = _pick("terminals", "dones")
        timeouts = _pick("timeouts", "truncateds")
        log_probs = _pick(
            "infos/action_log_probs",
            "infos__action_log_probs",
            "log_probs",
            "action_log_probs",
        )

        if observations is None or actions is None or rewards is None:
            raise KeyError(
                "Expert trajectories must contain observations/states, actions, and rewards."
            )
        if next_observations is None:
            raise KeyError("Expert trajectories must contain next_observations/next_states.")
        if terminals is None:
            terminals = np.zeros(total_length, dtype=np.float32)
        if timeouts is None:
            timeouts = np.zeros(total_length, dtype=np.float32)

        observations = np.asarray(observations, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)
        timeouts = np.asarray(timeouts, dtype=np.float32).reshape(-1)
        log_probs = (
            np.zeros(total_length, dtype=np.float32)
            if log_probs is None
            else np.asarray(log_probs, dtype=np.float32).reshape(-1)
        )

        self.absorbing = absorbing
        if absorbing:
            (observations, next_observations, actions, rewards, terminals,
             timeouts, log_probs, absorbing_flags, next_absorbing_flags,
             lengths) = self._wrap_with_absorbing(
                observations, next_observations, actions, rewards,
                terminals, timeouts, log_probs, lengths,
            )
            total_length = sum(lengths)
        else:
            absorbing_flags = np.zeros(total_length, dtype=np.float32)
            next_absorbing_flags = np.zeros(total_length, dtype=np.float32)

        self.episode_ends = np.cumsum(lengths) - 1

        super(ExpertDataset, self).__init__(
            observation_space, action_space, total_length, 1, gamma, nstep, onpolicy=True, store_u=False
        )
        if absorbing:
            for key in ("absorbing", "next_absorbing"):
                self.buffer_message[key] = {"shape": (1, total_length), "dtype": np.float32}
            self.reset()

        expected_obs_dim = self.buffer["states"].shape[-1]
        if observations.shape[-1] != expected_obs_dim:
            raise ValueError(
                f"Expert observations have {observations.shape[-1]} dimensions but the "
                f"observation space expects {expected_obs_dim}"
                + (" (absorbing adds one indicator dimension)" if absorbing else "")
            )

        self.buffer["states"][0] = observations
        self.buffer["next_states"][0] = next_observations
        self.buffer["actions"][0] = actions
        self.buffer["rewards"][0] = rewards
        self.buffer["dones"][0] = terminals
        self.buffer["truncateds"][0] = timeouts
        if "log_probs" in self.buffer:
            self.buffer["log_probs"][0] = log_probs
        if absorbing:
            self.buffer["absorbing"][0] = absorbing_flags
            self.buffer["next_absorbing"][0] = next_absorbing_flags

        self.pos = total_length % self.buffer_size
        self.full = total_length >= self.buffer_size

    @staticmethod
    def _wrap_with_absorbing(observations, next_observations, actions, rewards,
                             terminals, timeouts, log_probs, lengths):
        """Rewrite expert episodes in DAC's absorbing-state MDP.

        Every observation gains an indicator bit (0). Every real termination
        (not a time limit) is scanned — including those that are not the last
        step of a split trajectory — redirected to the absorbing state, and
        followed by an absorbing self-loop, matching the policy-side wrap.
        """
        obs_dim = observations.shape[-1] + 1
        absorbing_obs = absorbing_state(obs_dim)
        absorbing_act = absorbing_action(actions.shape[-1])

        parts = {key: [] for key in (
            "observations", "next_observations", "actions", "rewards",
            "terminals", "timeouts", "log_probs", "absorbing", "next_absorbing",
        )}
        new_lengths = []

        start = 0
        for length in lengths:
            episode = slice(start, start + length)
            start += length

            pad = np.zeros((length, 1), dtype=np.float32)
            wrapped = insert_absorbing_self_loops(
                np.concatenate([observations[episode], pad], axis=1),
                np.concatenate([next_observations[episode], pad], axis=1),
                actions[episode],
                rewards[episode],
                terminals[episode],
                timeouts[episode],
                absorbing_obs,
                absorbing_act,
                extras={"log_probs": (log_probs[episode], np.float32(0.0))},
            )
            parts["observations"].append(wrapped["states"])
            parts["next_observations"].append(wrapped["next_states"])
            parts["actions"].append(wrapped["actions"])
            parts["rewards"].append(wrapped["rewards"])
            parts["terminals"].append(wrapped["dones"])
            parts["timeouts"].append(wrapped["truncateds"])
            parts["log_probs"].append(wrapped["log_probs"])
            parts["absorbing"].append(wrapped["absorbing"])
            parts["next_absorbing"].append(wrapped["next_absorbing"])
            new_lengths.append(len(wrapped["rewards"]))

        merged = {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in parts.items()}
        return (merged["observations"], merged["next_observations"], merged["actions"],
                merged["rewards"], merged["terminals"], merged["timeouts"],
                merged["log_probs"], merged["absorbing"], merged["next_absorbing"],
                new_lengths)
        

    @staticmethod
    def _split_trajectories(data: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
        """Split flat D4RL-style arrays into per-episode dicts via terminals/timeouts."""
        if (
            "episode_ends" not in data
            and "terminals" not in data
            and "timeouts" not in data
        ):
            raise KeyError(
                "Expert dataset must contain 'episode_ends', "
                "'terminals', or 'timeouts' to recover episode boundaries."
            )

        n = data["actions"].shape[0]
        terminals = (
            np.asarray(data["terminals"], dtype=np.bool_).reshape(-1)
            if "terminals" in data
            else np.zeros(n, dtype=np.bool_)
        )
        timeouts = (
            np.asarray(data["timeouts"], dtype=np.bool_).reshape(-1)
            if "timeouts" in data
            else np.zeros(n, dtype=np.bool_)
        )
        if terminals.shape[0] != n or timeouts.shape[0] != n:
            raise ValueError("terminals/timeouts length does not match dataset length.")

        if "episode_ends" in data:
            episode_end_mask = np.asarray(
                data["episode_ends"],
                dtype=np.bool_,
            ).reshape(-1)

            if episode_end_mask.shape[0] != n:
                raise ValueError(
                    "episode_ends length does not match dataset length."
                )

            if episode_end_mask.size == 0 or not episode_end_mask[-1]:
                raise ValueError(
                    "The last transition must be marked as an episode end."
                )
        else:
            episode_end_mask = terminals | timeouts

        episode_ends = np.flatnonzero(episode_end_mask)
        
        trajectories: list[dict[str, np.ndarray]] = []
        start = 0
        for end in episode_ends:
            if end < start:
                continue
            trajectories.append({key: value[start : end + 1] for key, value in data.items() if key != "episode_ends"})
            start = end + 1
        # leftover steps without a terminal/timeout flag
        if start < n:
            trajectories.append({key: value[start:] for key, value in data.items() if key != "episode_ends"})
        return trajectories




if __name__ == "__main__":
    expert_dataset = ExpertDataset("/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/sac_hopper_10eps.hdf5",
                                   observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,)),
                                   action_space=gym.spaces.Box(low=-1, high=1, shape=(3,)),
                                   trajectory_num=10,
                                   subsample_frequency=1,
                                   gamma=0.99,
                                   nstep=1)
    print(expert_dataset.trajectories)
