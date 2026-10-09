"""HyPE in the DAC/SAC interfaces used by this library.

Reference: outputs/garage/garage/algorithms/model_free_irl.py.
"""
import numpy as np
import torch
from omegaconf import OmegaConf

from .dac import DAC
from .sac import SAC
from env.make_envs import make_vec_envs
from utils import OPTIMIZER_DICT
from utils.result import Result
from utils.utils import to_useful_action


class HyPE(DAC):
    """Hybrid SAC sampling and moment-matching cost learning from fresh rollouts.

    DAC supplies transition handling, gradient penalties and reward relabelling.
    HyPE changes the sampling distribution, discriminator objective and updates
    the discriminator after policy training, using the current policy's rollouts.
    """

    def __init__(self, training_envs, testing_envs, buffer, expert_buffer, agent,
                 discriminator, logger, device, args, rl_args, discriminator_args):
        super().__init__(training_envs, testing_envs, buffer, expert_buffer, agent,
                         discriminator, logger, device, args, rl_args, discriminator_args)
        if buffer.nstep != 1 or expert_buffer.nstep != 1:
            raise ValueError("HyPE hybrid sampling currently requires nstep=1")
        self.sampling_schedule = np.asarray(args.algorithm.sampling_schedule, dtype=np.float64)
        schedule = self.sampling_schedule
        if (schedule.ndim != 2 or schedule.shape[1] != 3 or len(schedule) == 0
                or not np.isfinite(schedule).all()
                or np.any(schedule[:, :2] < 0) or np.any(schedule[:, :2] > 1)
                or np.any(np.diff(np.r_[0, schedule[:, 2]]) <= 0)):
            raise ValueError("Expected [start_ratio, end_ratio, increasing_end_step] rows")
        self.policy_lr_decay = bool(args.algorithm.policy_lr_decay)
        self.reset_policy_timesteps = bool(getattr(args.algorithm, "reset_policy_timesteps", False))
        self.discriminator_rollout_episodes = int(discriminator_args.rollout_episodes)
        if self.discriminator_rollout_episodes < 1 or self.discriminator_train_interval < 1:
            raise ValueError("HyPE rollout_episodes and discriminator_train_interval must be positive")
        if min(self.update_step_per_epoch, self.update_discriminator_step_per_epoch,
               self.discriminator_train_steps, self.discriminator_batch_size) < 1:
            raise ValueError("HyPE update counts and discriminator_batch_size must be positive")
        self.discriminator_args = discriminator_args
        self.discriminator_envs = None
        self.discriminator_samples = None
        self.discriminator_round = 0
        self.policy_interaction_step = 0
        self._policy_phase_start_step = self.interaction_step

    def _policy_phase_step(self):
        # During policy collection this includes the current, not-yet-buffered
        # transitions. Reset after D rollouts so they never consume warmup.
        return self.interaction_step - self._policy_phase_start_step

    def random_choose_action(self):
        if self.reset_policy_timesteps:
            return self._policy_phase_step() < self.start_train_step
        return super().random_choose_action()

    def start_train(self):
        if self.reset_policy_timesteps:
            # SB3 starts gradient updates strictly after learning_starts.
            return self._policy_phase_step() > self.start_train_step
        return super().start_train()

    def update(self, batch, start_train):
        self.policy_interaction_step += int(np.prod(batch['states'].shape[:2]))
        result = self._update_buffer(batch)
        if not start_train and not self.reset_policy_timesteps:
            return result

        if start_train:
            for _ in range(self.update_step_per_epoch):
                policy_result = self._update_policy()
            result.add(policy_result)
        if self.reset_policy_timesteps:
            result.add_metric('hybrid/policy_phase_step', self._policy_phase_step())

        # As elsewhere in this library, the interval is measured in epochs.
        # A reset-mode phase still finishes when its whole budget is warmup.
        if (self.current_epoch + 1) % self.discriminator_train_interval == 0:
            rollout_steps = self._collect_discriminator_samples()
            # garage starts a fresh OAdam each round and uses lr / max(1, round).
            disc_lr = self.disc_lr / max(1, self.discriminator_round)
            self.discriminator_optimizer = OPTIMIZER_DICT[
                self.discriminator_args.discriminator_optimizer](
                    self.discriminator.parameters(), lr=disc_lr,
                    weight_decay=self.discriminator_args.discriminator_weight_decay)
            for _ in range(self.update_discriminator_step_per_epoch):
                disc_result = self._update_discriminator()
            self.discriminator_round += 1
            disc_result.add_metric('discriminator/lr', disc_lr)
            disc_result.add_metric('discriminator/round', self.discriminator_round)
            disc_result.add_metric('discriminator/rollout_steps', rollout_steps)
            result.add(disc_result)
            if self.reset_policy_timesteps:
                self._policy_phase_start_step = self.interaction_step
        result.add_metric('hybrid/policy_interaction_step', self.policy_interaction_step)
        return result

    def _collect_discriminator_samples(self):
        """Collect current-policy samples without resetting the training env.

        These samples train the cost only, matching garage. Their interactions
        are included in the total interaction_step reported by the library.
        """
        if self.discriminator_envs is None:
            env_args = OmegaConf.create(OmegaConf.to_container(self.args.env, resolve=True))
            env_args.num_testing_envs = 1
            self.discriminator_envs = make_vec_envs(env_args, False, seed=self.seed + 10000)
        envs = self.discriminator_envs
        if self.args.env.obs_norm:
            envs.set_obs_rms(self.training_envs.get_obs_rms())
        states, actions = [], []
        completed, steps = 0, 0
        was_training = self.agent.training
        self.agent.eval()
        try:
            observations = envs.reset()
            while completed < self.discriminator_rollout_episodes:
                with torch.no_grad():
                    sampled_actions, _ = self.agent.select_action(observations, deterministic=False)
                sampled_actions = to_useful_action(self.action_space, self.action_dim, sampled_actions)
                states.append(np.asarray(observations[0], dtype=np.float32).copy())
                actions.append(np.asarray(sampled_actions[0], dtype=np.float32).copy())
                observations, _, dones, infos = envs.step(sampled_actions)
                steps += 1
                self.interaction_step += 1
                if dones[0]:
                    completed += 1
                    if self.absorbing and not infos[0].get('TimeLimit.truncated', False) and not infos[0].get('truncated', False):
                        states.append(self.absorbing_state.copy())
                        actions.append(self.absorbing_action.copy())
        finally:
            self.agent.train(was_training)
        self.discriminator_samples = dict(states=np.stack(states), actions=np.stack(actions))
        return steps

    def _update_discriminator(self):
        if self.discriminator_samples is None:
            raise RuntimeError("Collect current-policy trajectories before training the HyPE cost")
        self.discriminator.train()
        with Result('discriminator') as result:
            for _ in range(self.discriminator_train_steps):
                indices = np.random.randint(len(self.discriminator_samples['states']),
                                            size=self.discriminator_batch_size)
                policy_states = torch.as_tensor(self.discriminator_samples['states'][indices],
                                                dtype=torch.float32, device=self.device)
                policy_actions = torch.as_tensor(self.discriminator_samples['actions'][indices],
                                                 dtype=torch.float32, device=self.device)
                expert = self.expert_buffer.sample(self.discriminator_batch_size)
                expert_states = self._expert_states_to_tensor(expert['states'], expert.get('absorbing'))
                expert_actions = torch.as_tensor(expert['actions'], dtype=torch.float32, device=self.device)
                expert_cost = self.discriminator.predict_logits(torch.cat([expert_states, expert_actions], dim=1))
                policy_cost = self.discriminator.predict_logits(torch.cat([policy_states, policy_actions], dim=1))
                loss = expert_cost.mean() - policy_cost.mean()
                penalty = loss.new_zeros(())
                if self.discriminator_gradient_penalty:
                    penalty = self._compute_gradient_penalty(expert_states, expert_actions,
                                                            policy_states, policy_actions)
                loss = loss + penalty
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()
        result.add_metric('discriminator/loss', loss.item())
        result.add_metric('discriminator/expert_cost', expert_cost.mean().item())
        result.add_metric('discriminator/policy_cost', policy_cost.mean().item())
        result.add_metric('discriminator/gradient_penalty', penalty.item())
        return result

    def _expert_sampling_ratio(self, step=None):
        """Evaluate the schedule using the number of completed SAC updates."""
        step = self.gradient_step if step is None else step
        start_step = 0
        for start_ratio, end_ratio, end_step in self.sampling_schedule:
            if step <= end_step:
                fraction = np.clip((step - start_step) / (end_step - start_step), 0, 1)
                return float(start_ratio + fraction * (end_ratio - start_ratio))
            start_step = end_step
        return float(self.sampling_schedule[-1, 1])

    def _update_policy(self):
        expert_ratio = self._expert_sampling_ratio()
        batch = self.buffer.sample_mixed(self.batch_size, self.expert_buffer, expert_ratio)
        expert_mask = batch.pop('expert_mask')
        if expert_mask.any():
            for key, flag in (('states', 'absorbing'), ('next_states', 'next_absorbing')):
                batch[key][expert_mask] = self._expert_states_to_tensor(
                    batch[key][expert_mask], batch[flag][expert_mask]).detach().cpu().numpy()
            batch['nstep_states'][expert_mask, 0] = batch['states'][expert_mask]
        batch['rewards'] = self._nstep_discriminator_rewards(batch)

        # Only the policy clock is optional. Gradient steps, expert sampling,
        # model/optimizer state and the total training budget stay continuous.
        if self.policy_lr_decay:
            if self.reset_policy_timesteps:
                steps = self._policy_phase_step()
                epochs = self.discriminator_train_interval
            else:
                steps = self.policy_interaction_step
                epochs = self.total_epoch
            budget = epochs * self.interact_per_epoch * self.num_training_envs
            fraction = max(0.0, 1.0 - steps / max(1, budget))
            optimizers = [(self.actor_optimizer, self.actor_lr),
                          (self.critic_optimizer, self.critic_lr)]
            if self.learn_temp:
                optimizers.append((self.temp_optimizer, self.temp_lr))
            for optimizer, initial_lr in optimizers:
                for group in optimizer.param_groups:
                    group['lr'] = initial_lr * fraction
        result = self._sac_update(batch)
        result.add_metric('discriminator/reward_mean', float(batch['rewards'].mean()))
        result.add_metric('hybrid/expert_ratio', expert_ratio)
        result.add_metric('hybrid/expert_fraction', float(expert_mask.mean()))
        result.add_metric('hybrid/sample_step', self.gradient_step)
        result.add_metric('actor/lr', self.actor_optimizer.param_groups[0]['lr'])
        return result

    def run(self):
        try:
            super().run()
        finally:
            if self.discriminator_envs is not None:
                self.discriminator_envs.close()


class HyPESAC(HyPE, SAC):
    pass
