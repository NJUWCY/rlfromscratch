# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Primal Wasserstein imitation rewards with the library's SAC learner.

Algorithm 1 / Eq. (6), Dadashi et al., ICLR 2021. The reference in
outputs/pwil uses D4PG; this module reuses SAC, ReplayBuffer and ExpertDataset.
"""

import copy
from collections import deque

import numpy as np
from sklearn import preprocessing
from stable_baselines3.common.vec_env import VecEnvWrapper

from env.make_envs import VecObsNorm
from utils.utils import RunningMeanStd
from .sac import SAC


class PWILRewarder:
    """Reference PWIL rewards, with independent mutable atoms per environment."""

    def __init__(self, observations, actions, time_horizon=1000, alpha=5.0,
                 beta=5.0, observation_only=False, num_envs=1):
        # Preserve the supplied precision, just like the reference vectorize().
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        if (observations.ndim != 2 or actions.ndim != 2
                or len(observations) == 0 or len(observations) != len(actions)
                or not np.isfinite(observations).all() or not np.isfinite(actions).all()):
            raise ValueError("PWIL requires nonempty, finite, paired vector observations/actions")
        if (int(time_horizon) != time_horizon or time_horizon < 1
                or int(num_envs) != num_envs or num_envs < 1):
            raise ValueError("PWIL time_horizon and num_envs must be positive integers")
        if not np.isfinite([alpha, beta]).all() or alpha <= 0 or beta <= 0:
            raise ValueError("PWIL alpha and beta must be finite and positive")
        self.time_horizon = int(time_horizon)
        self.num_envs = int(num_envs)
        self.alpha, self.beta = float(alpha), float(beta)
        self.observation_only = bool(observation_only)
        self.obs_dim, self.action_dim = observations.shape[1], actions.shape[1]
        features = observations if observation_only else np.concatenate([observations, actions], axis=1)
        self.vectorized_demonstrations = features.copy()
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(self.vectorized_demonstrations)
        self.num_atoms = len(features)
        # Retain sqrt(|S|+|A|) also in observation-only mode, as in the reference.
        self.reward_sigma = self.beta * self.time_horizon / np.sqrt(self.obs_dim + self.action_dim)
        # Each stream deletes exhausted atoms independently, as the reference
        # does for its single environment. num_atoms remains the original size.
        self.expert_atoms = [None] * self.num_envs
        self.expert_weights = [None] * self.num_envs
        self.steps = np.zeros(self.num_envs, dtype=np.int64)
        self.last_cost = np.zeros(self.num_envs, dtype=np.float64)
        self.reset()

    def reset(self, env_index=None):
        indices = range(self.num_envs) if env_index is None else (env_index,)
        for i in indices:
            self.expert_atoms[i] = copy.deepcopy(
                self.scaler.transform(self.vectorized_demonstrations))
            self.expert_weights[i] = np.ones(self.num_atoms) / self.num_atoms
            self.steps[i] = 0
            self.last_cost[i] = 0

    def compute_reward(self, observation, action, env_index=0):
        observation = np.asarray(observation)
        action = np.asarray(action)
        if observation.shape != (self.obs_dim,) or action.shape != (self.action_dim,):
            raise ValueError("PWIL observation/action dimensions do not match expert atoms")
        atom = observation if self.observation_only else np.concatenate([observation, action])
        if not np.isfinite(atom).all():
            raise ValueError("PWIL received a non-finite policy state/action")
        agent_atom = self.scaler.transform(np.expand_dims(atom, axis=0))[0]

        # Keep the reference's floating-point operation order and epsilon.
        cost = 0.
        weight = 1. / self.time_horizon - 1e-6
        norms = np.linalg.norm(self.expert_atoms[env_index] - agent_atom, axis=1)
        while weight > 0:
            argmin = norms.argmin()
            expert_weight = self.expert_weights[env_index][argmin]
            if weight >= expert_weight:
                weight -= expert_weight
                cost += expert_weight * norms[argmin]
                self.expert_weights[env_index] = np.delete(
                    self.expert_weights[env_index], argmin, 0)
                self.expert_atoms[env_index] = np.delete(
                    self.expert_atoms[env_index], argmin, 0)
                norms = np.delete(norms, argmin, 0)
            else:
                cost += weight * norms[argmin]
                self.expert_weights[env_index][argmin] -= weight
                weight = 0

        self.steps[env_index] += 1
        self.last_cost[env_index] = cost
        reward = self.alpha * np.exp(-self.reward_sigma * cost)
        return reward.astype('float32')


class PWILRewardWrapper(VecEnvWrapper):
    """Replace training rewards before observation normalization and replay.

    The underlying Monitor still records true environment returns. SB3's done
    includes termination and timeout; either resets only that environment's mass.
    """

    def __init__(self, envs, rewarder, reward_buffer_size=20):
        super().__init__(envs)
        if envs.num_envs != rewarder.num_envs:
            raise ValueError("PWIL needs one independent mass vector per environment")
        self.rewarder = rewarder
        self.raw_observations = None
        self.actions = None
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.completed_returns = deque(maxlen=reward_buffer_size)

    def reset(self):
        self.rewarder.reset()
        self.episode_returns.fill(0)
        self.raw_observations = self.venv.reset().copy()
        return self.raw_observations.copy()

    def step_async(self, actions):
        self.actions = np.asarray(actions).copy()
        self.venv.step_async(actions)

    def step_wait(self):
        observations, task_rewards, dones, infos = self.venv.step_wait()
        rewards = np.empty(self.num_envs, dtype=np.float32)
        for i in range(self.num_envs):
            # Official load_environment() uses Acme's SinglePrecisionWrapper.
            rewards[i] = self.rewarder.compute_reward(
                np.asarray(self.raw_observations[i], dtype=np.float32),
                np.asarray(self.actions[i], dtype=np.float32), i)
            self.episode_returns[i] += float(rewards[i])
            infos[i]["pwil_cost"] = float(self.rewarder.last_cost[i])
            infos[i]["task_reward"] = float(task_rewards[i])
            if dones[i]:
                self.completed_returns.append(float(self.episode_returns[i]))
                self.episode_returns[i] = 0
                self.rewarder.reset(i)
        # SB3 returns the next episode's reset observation after done.
        self.raw_observations = observations.copy()
        return observations, rewards, dones, infos


class PWIL(SAC):
    """PWIL-SAC: collection-time greedy rewards, demonstration-prefilled replay."""

    def __init__(self, training_envs, testing_envs, buffer, expert_buffer,
                 agent, logger, device, args, rl_args):
        if args.env.absorbing or getattr(expert_buffer, "absorbing", False):
            raise ValueError("PWIL requires env.absorbing=false")
        if int(rl_args.nstep) != 1 or buffer.nstep != 1 or expert_buffer.nstep != 1:
            raise ValueError("PWIL-SAC currently supports nstep=1")
        if isinstance(training_envs, VecObsNorm):
            raise ValueError("Pass raw environments; PWIL installs observation normalization after its reward wrapper")
        if buffer is expert_buffer or expert_buffer.num_envs != 1:
            raise ValueError("PWIL requires a separate, single-stream ExpertDataset")
        pwil_args = args.algorithm
        horizon = int(pwil_args.time_horizon)
        if horizon < int(args.env.max_episode_length):
            raise ValueError("PWIL time_horizon must cover env.max_episode_length")
        # ExpertDataset already selected/subsampled the demonstrations.
        # Use every stored transition for both reward atoms and replay prefill.
        self.expert_buffer = expert_buffer
        expert_states = expert_buffer.buffer["states"][0]
        expert_actions = expert_buffer.buffer["actions"][0]
        self.rewarder = PWILRewarder(
            expert_states, expert_actions, horizon,
            alpha=pwil_args.alpha, beta=pwil_args.beta,
            observation_only=pwil_args.observation_only,
            num_envs=training_envs.num_envs,
        )
        self.reward_envs = PWILRewardWrapper(training_envs, self.rewarder, args.reward_buffer_size)
        policy_envs = self.reward_envs

        self.num_transitions_rb = int(pwil_args.num_transitions_rb)
        if self.num_transitions_rb < 0 or self.num_transitions_rb > buffer.buffer_size * buffer.num_envs:
            raise ValueError("PWIL num_transitions_rb must lie within total replay capacity")
        self.prefilled_transitions = 0
        self._prefilled = False
        super().__init__(policy_envs, testing_envs, buffer, agent, logger, device, args, rl_args)

    def initialize(self):
        if not self._prefilled:
            # The vector replay has equal capacity on each stream. Round down by
            # at most num_envs-1 transitions; default 50000 divides 1/2/4/8 evenly.
            per_env = self.num_transitions_rb // self.num_training_envs
            if per_env > 0:
                positions = np.arange(per_env)[None, :]
                positions = positions + np.arange(self.num_training_envs)[:, None] * per_env
                indices = positions % self.rewarder.num_atoms
                batch = {key: self.expert_buffer.buffer[key][0, indices].copy()
                         for key in self.buffer.buffer}
                batch["rewards"].fill(self.rewarder.alpha)
                if self.args.env.obs_norm:
                    for key in ("states", "next_states"):
                        batch[key] = self.training_envs._norm_obs(batch[key]).astype(np.float32)
                # Use the stored real one-step next_state, never the next
                # subsampled demonstration state, and preserve done/timeout.
                self.buffer.add(batch)
            self.prefilled_transitions = per_env * self.num_training_envs
            self._prefilled = True
        super().initialize()

    def update(self, batch, start_train):
        # Rewards are already PWIL rewards. Do not relabel shuffled replay data.
        result = super().update(batch, start_train)
        result.add_metric("pwil/reward_mean", float(batch["rewards"].mean()))
        result.add_metric("pwil/reward_min", float(batch["rewards"].min()))
        result.add_metric("pwil/reward_max", float(batch["rewards"].max()))
        result.add_metric("pwil/expert_atoms", self.rewarder.num_atoms)
        result.add_metric("pwil/prefilled_transitions", self.prefilled_transitions)
        result.add_metric("pwil/costs", float(self.rewarder.last_cost.mean()))
        if self.reward_envs.completed_returns:
            result.add_metric("pwil/episode_return_mean", float(np.mean(self.reward_envs.completed_returns)))
        return result
