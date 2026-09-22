import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .baseailalgorithm import AILAlgorithm
from .baseoffpolicy import OffPolicyAlgorithm
from .sac import SAC
from memory.memory import AbsorbingReplayBuffer
from env.absorbing import absorbing_state, absorbing_action, insert_absorbing_self_loops
from utils.result import Result
from logger.logger import Logger
from agent.agent import AgentBase
from discriminator.discriminator import DiscriminatorBase
from utils import OPTIMIZER_DICT


class DAC(AILAlgorithm, OffPolicyAlgorithm):

    """Discriminator-Actor-Critic (Kostrikov et al., 2019).

    Off-policy adversarial imitation learning. Three things distinguish it from
    the on-policy GAIL in this repository:

    1. The discriminator is trained against samples drawn from the whole replay
       buffer instead of the freshly collected rollout.
    2. Rewards are never stored. They are recomputed by the current
       discriminator on every sampled minibatch, since a stored reward would be
       stale as soon as the discriminator moves.
    3. Episode terminations are redirected into an explicit absorbing state, so
       that the discriminator cannot infer "expert vs policy" from episode
       length alone, and the critic learns the value of terminating.
    """

    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: AbsorbingReplayBuffer, expert_buffer, agent: AgentBase, discriminator:DiscriminatorBase, logger: Logger, device, args: DictConfig, rl_args: DictConfig, discriminator_args: DictConfig):
        super(DAC, self).__init__(training_envs, testing_envs, buffer, expert_buffer, agent, discriminator, logger, device, args, rl_args, discriminator_args)

        self.disc_lr = discriminator_args.disc_lr
        self.gradient_penalty_coef = discriminator_args.gradient_penalty_coef
        self.discriminator_gradient_penalty = discriminator_args.discriminator_gradient_penalty
        self.discriminator_train_steps = discriminator_args.discriminator_train_steps
        self.discriminator_batch_size = discriminator_args.discriminator_batch_size

        self.discriminator_optimizer = OPTIMIZER_DICT[discriminator_args.discriminator_optimizer](self.discriminator.parameters(), lr=self.disc_lr)

        self.absorbing = bool(getattr(args.env, "absorbing", False))
        self.absorbing_state = absorbing_state(self.observation_space.shape[0])
        self.absorbing_action = absorbing_action(self.action_dim)
        self.absorbing_transitions = 0

    def update(self, batch, start_train) -> Result:
        result = self._update_buffer(batch)

        if start_train:
            for _ in range(self.update_discriminator_step_per_epoch):
                discriminator_result = self._update_discriminator()
            result.add(discriminator_result)

            for _ in range(self.update_step_per_epoch):
                update_policy_log = self._update_policy()
            result.add(update_policy_log)

        return result

    def _wrap_absorbing_sequence(self, states, next_states, actions, rewards, dones, truncateds) -> dict:
        """Insert absorbing self-loops immediately after every real termination.

        A real termination ``(s, a, r, s_T)`` becomes ``(s, a, r, s_absorbing)``
        with ``done = 0``, followed by an extra ``(s_absorbing, 0, 0,
        s_absorbing)`` self-loop sitting in the next time index. Time limits
        are left alone: they are not a property of the MDP and must stay
        bootstrapped.
        """
        if not self.absorbing:
            states = np.asarray(states, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
            length = len(rewards)
            return {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": np.asarray(next_states, dtype=np.float32),
                "dones": np.asarray(dones, dtype=np.float32).reshape(-1),
                "truncateds": np.asarray(truncateds, dtype=np.float32).reshape(-1),
                "absorbing": np.zeros(length, dtype=np.float32),
                "next_absorbing": np.zeros(length, dtype=np.float32),
            }
        return insert_absorbing_self_loops(
            states, next_states, actions, rewards, dones, truncateds,
            self.absorbing_state, self.absorbing_action,
        )

    def _update_buffer(self, batch):
        with Result("buffer") as result:
            num_envs = batch['states'].shape[0]
            for env_i in range(num_envs):
                transitions = self._wrap_absorbing_sequence(
                    batch['states'][env_i],
                    batch['next_states'][env_i],
                    batch['actions'][env_i],
                    batch['rewards'][env_i],
                    batch['dones'][env_i],
                    batch['truncateds'][env_i],
                )
                self.buffer.add_transitions(transitions, env_index=env_i)
                self.absorbing_transitions += int(transitions['absorbing'].sum())
        result.add_metric("buffer/size", float(self.buffer.filled()))
        result.add_metric("buffer/absorbing_transitions", float(self.absorbing_transitions))
        return result

    def _expert_states_to_tensor(self, states: np.ndarray, absorbing_flags: np.ndarray) -> torch.Tensor:
        """Put expert observations in the space the policy observations live in.

        Policy observations are normalised by the vectorised env before they
        reach the buffer, whereas absorbing states are written afterwards and
        stay the raw constant. Expert data is stored raw, so it is normalised
        here and its absorbing rows are then restored to that same constant: if
        the two sides disagreed on what "absorbing" looks like, the
        discriminator could tell expert from policy on that cue alone.
        """
        states = np.asarray(states, dtype=np.float32)
        if not self.args.env.obs_norm:
            return torch.from_numpy(states).to(self.device)

        states = np.asarray(self.training_envs._norm_obs(states), dtype=np.float32)
        if absorbing_flags is not None:
            mask = np.asarray(absorbing_flags) > 0
            if mask.any():
                states[mask] = self.absorbing_state
        return torch.from_numpy(states).to(self.device)

    def _compute_gradient_penalty(self, expert_states:torch.Tensor, expert_actions:torch.Tensor, policy_states:torch.Tensor, policy_actions:torch.Tensor):
        alpha = torch.rand(expert_states.shape[0], 1, device=expert_states.device)
        expert_data = torch.cat([expert_states, expert_actions], dim=1)
        policy_data = torch.cat([policy_states, policy_actions], dim=1)
        mixed_data = alpha*expert_data + (1-alpha)*policy_data
        mixed_data.requires_grad = True
        mixed_logits = self.discriminator.predict_logits(mixed_data)

        gradient = torch.autograd.grad(outputs=mixed_logits.sum(), inputs=mixed_data, create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = self.gradient_penalty_coef*((gradient_norm - 1)**2).mean()
        return gradient_penalty

    def _update_discriminator(self):
        with Result("discriminator") as result:
            for _ in range(self.discriminator_train_steps):
                policy_batch = self.buffer.sample(self.discriminator_batch_size)
                expert_batch = self.expert_buffer.sample(self.discriminator_batch_size)

                policy_states = torch.from_numpy(np.asarray(policy_batch['states'], dtype=np.float32)).to(self.device)
                policy_actions = torch.from_numpy(np.asarray(policy_batch['actions'], dtype=np.float32)).to(self.device)
                expert_states = self._expert_states_to_tensor(expert_batch['states'], expert_batch.get('absorbing'))
                expert_actions = torch.from_numpy(np.asarray(expert_batch['actions'], dtype=np.float32)).to(self.device)

                expert_logits = self.discriminator.predict_logits(torch.cat([expert_states, expert_actions], dim=1))
                policy_logits = self.discriminator.predict_logits(torch.cat([policy_states, policy_actions], dim=1))

                loss = nn.functional.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits)) \
                    + nn.functional.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))

                if self.discriminator_gradient_penalty:
                    loss = loss + self._compute_gradient_penalty(expert_states, expert_actions, policy_states, policy_actions)

                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()

        result.add_metric("discriminator/loss", loss.item())
        result.add_metric("discriminator/expert_D", nn.functional.sigmoid(expert_logits).mean().item())
        result.add_metric("discriminator/policy_D", nn.functional.sigmoid(policy_logits).mean().item())
        return result

    def _nstep_discriminator_rewards(self, batch: dict) -> np.ndarray:
        """Sum current discriminator rewards along the n-step window.

        Stored environment rewards are stale for DAC even at nstep=1; the
        discriminator has to score every (s, a) that the n-step return uses.
        """
        states = np.asarray(batch['nstep_states'], dtype=np.float32)
        actions = np.asarray(batch['nstep_actions'], dtype=np.float32)
        mask = np.asarray(batch['nstep_mask'], dtype=np.float32)
        batch_size, nstep = mask.shape
        rewards = self.discriminator.predict_reward(
            states.reshape(batch_size * nstep, -1),
            actions.reshape(batch_size * nstep, -1),
        )
        rewards = np.asarray(rewards, dtype=np.float32).reshape(batch_size, nstep)
        discounts = self.gamma ** np.arange(nstep, dtype=np.float32)
        return (rewards * mask * discounts).sum(axis=1)

    def _update_policy(self):
        batch = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            rewards = self._nstep_discriminator_rewards(batch)
        batch['rewards'] = np.asarray(rewards, dtype=np.float32).reshape(-1)

        result = self._sac_update(batch)
        result.add_metric("discriminator/reward_mean", float(batch['rewards'].mean()))
        return result


class DACSAC(DAC, SAC):
    pass
