"""Continuous-control IQ-Learn using the existing SAC agent and replay.

Adapted from /home/ubuntu/wangchenyang/iq_learn/iq.py and train_iq.py.
Copyright (2021), Stanford University; Copyright 2022 Div Garg.
See IQ_LEARN_LICENSE.txt for the reference implementation's license.
"""
import math
from collections import deque
import numpy as np
import torch
from .sac import SAC
from utils.result import Result


class IQLearn(SAC):
    """Online inverse soft-Q learning; task rewards are only used for evaluation."""

    def __init__(self, training_envs, testing_envs, buffer, expert_buffer,
                 agent, logger, device, args, rl_args):
        if args.env.absorbing or int(rl_args.nstep) != 1:
            raise ValueError("IQ-Learn requires env.absorbing=false and nstep=1")
        if buffer.nstep != 1 or expert_buffer.nstep != 1 or buffer is expert_buffer:
            raise ValueError("IQ-Learn requires separate one-step expert and online buffers")
        if expert_buffer.buffer_size <= 0 or (not expert_buffer.full and expert_buffer.pos == 0):
            raise ValueError("IQ-Learn requires a nonempty expert dataset")
        cfg = args.algorithm
        self.iq_loss_type = cfg.iq_loss_type
        self.v0_mix_type = cfg.v0_mix_type
        self.divergence = cfg.div
        self.chi = bool(cfg.chi)
        self.regularize = bool(cfg.regularize)
        self.iq_alpha = float(cfg.alpha)
        self.actor_update_interval = int(cfg.actor_update_interval)
        self.num_actor_updates = int(cfg.num_actor_updates)
        self.next_v_use_target = bool(cfg.next_v_use_target)
        self.penalty = float(cfg.penalty)
        self.left = float(cfg.left)
        self.right = float(cfg.right)
        if not (math.isfinite(self.left) and math.isfinite(self.right)) or self.left > self.right:
            raise ValueError("algorithm.left/right must be finite and left <= right")
        if not math.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError("algorithm.penalty must be finite and nonnegative")
        if self.iq_loss_type not in {"value", "v0"}:
            raise ValueError("iq_loss_type must be value or v0")
        if self.v0_mix_type not in {"expert", "online", "mix"}:
            raise ValueError("v0_mix_type must be expert, online or mix")
        if self.divergence not in {None, "chi", "hellinger", "kl", "kl2", "kl_fix", "js"}:
            raise ValueError("Unsupported IQ divergence")
        if not math.isfinite(self.iq_alpha) or self.iq_alpha <= 0:
            raise ValueError("algorithm.alpha must be finite and positive")
        if min(int(rl_args.batch_size), self.actor_update_interval, self.num_actor_updates,
               int(rl_args.target_update_interval)) < 1:
            raise ValueError("Batch size and update intervals/counts must be positive")
        super().__init__(training_envs, testing_envs, buffer, agent, logger, device, args, rl_args)
        self.expert_buffer = expert_buffer
        if self.iq_loss_type == "v0" and self.v0_mix_type != "online":
            initial_states = getattr(expert_buffer, "initial_states", None)
            if (initial_states is None or len(initial_states) == 0
                    or not np.isfinite(initial_states).all()):
                raise ValueError("v0 requires finite pre-subsampling expert initial states")
        self.collect_init_states = bool(getattr(args.env, "collect_init_states", False))
        self.online_initial_states = deque(maxlen=cfg.online_init_buffersize)
        if self.iq_loss_type == "v0" and self.v0_mix_type != "expert" and not self.collect_init_states:
            raise ValueError("Online/mixed v0 requires env.collect_init_states=true")

    def _collect_initial_states(self):
        if self.collect_init_states:
            # One IPC transfer per collection phase, not per gradient update.
            # env_method also reaches Monitor through wrappers and subprocesses.
            for states in self.training_envs.env_method("pop_initial_states"):# Implemented in SB3 to apply pop_initial_states function to each environment
                self.online_initial_states.extend(states)

    def initialize(self):
        super().initialize()
        self._collect_initial_states()

    def interact_with_envs(self):
        batch, result = super().interact_with_envs()
        self._collect_initial_states()
        return batch, result

    def _sample_initial_states(self):
        pools = []
        if self.v0_mix_type in {"expert", "mix"}:
            pools.append(self.expert_buffer.initial_states)
        if self.v0_mix_type in {"online", "mix"}:
            pools.append(self.online_initial_states)
        parts = []
        for pool in pools:
            if not len(pool):
                raise RuntimeError("No online s0 collected; initialize/reset before updating IQ-Learn")
            indices = np.random.randint(len(pool), size=self.batch_size)
            parts.append(np.stack([pool[int(i)] for i in indices]))
        # mix contains exactly B expert + B online starts, independent of pool sizes.
        states = np.concatenate(parts).astype(np.float32, copy=False)
        if self.args.env.obs_norm:
            states = self.training_envs._norm_obs(states)
        return torch.as_tensor(states, dtype=torch.float32, device=self.device)

    def _value_loss(self, value, bootstrap, expert):
        if self.iq_loss_type == "v0":
            initial_value = self.agent.get_value(
                self._sample_initial_states(), temperature=self.log_temp.detach().exp(),
                use_target=False, action_grad=False,
            )
            return (1 - self.gamma) * initial_value.mean()
        residual = value - bootstrap
        if self.v0_mix_type == "expert":
            return residual[expert].mean()
        if self.v0_mix_type == "online":
            return residual[~expert].mean()
        # Explicit equal weighting, also robust to unequal input batch sizes.
        return 0.5 * (residual[expert].mean() + residual[~expert].mean())

    def _iq_loss(self, q, value, next_value, dones, expert):
        bootstrap = self.gamma * (1 - dones) * next_value
        reward = q - bootstrap
        expert_reward = reward[expert]
        # These weights are deliberately detached, matching the reference dual.
        with torch.no_grad():
            if self.divergence == "hellinger":
                weight = (1 + expert_reward).pow(-2)
            elif self.divergence == "kl":
                weight = torch.exp(-expert_reward - 1)
            elif self.divergence == "kl2":
                weight = torch.softmax(-expert_reward, dim=0) * len(expert_reward)
            elif self.divergence == "kl_fix":
                weight = torch.exp(-expert_reward)
            elif self.divergence == "js":
                exp_reward = torch.exp(-expert_reward)
                weight = exp_reward / (2 - exp_reward)
            else:
                weight = 1
        softq_loss = -(weight * expert_reward).mean()
        value_loss = self._value_loss(value, bootstrap, expert)
        chi_loss = q.new_zeros(())
        regularize_loss = q.new_zeros(())
        bellman_constraints = q - bootstrap.detach()
        if self.chi or self.divergence == "chi":
            chi_loss = bellman_constraints[expert].square().mean() / (4 * self.iq_alpha)
        if self.regularize:
            # Squared distance to [left, right]; no penalty inside the interval.
            regularize_loss = self.penalty * (
                torch.relu(self.left - bellman_constraints).square()
                + torch.relu(bellman_constraints - self.right).square()
            ).mean()
        loss = softq_loss + value_loss + chi_loss + regularize_loss
        return loss, dict(softq_loss=softq_loss, value_loss=value_loss,
                          chi2_loss=chi_loss, regularize_loss=regularize_loss,
                          expert_reward=reward[expert].mean(),
                          policy_reward=reward[~expert].mean())

    def _update_policy(self):
        # Reference batch_size is PER source: B online + B expert = 2B total.
        batch = self.buffer.sample_mixed(2 * self.batch_size, self.expert_buffer, 0.5)
        expert_mask = batch.pop("expert_mask")
        if self.args.env.obs_norm:
            for key in ("states", "next_states"):
                batch[key][expert_mask] = self.training_envs._norm_obs(batch[key][expert_mask])
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device).reshape(-1, 1)
        expert = torch.as_tensor(expert_mask, dtype=torch.bool, device=self.device)
        self.agent.train()
        with Result("train") as result:
            # v0 uses separately sampled initial states, not this batch's V(s).
            value = None
            if self.iq_loss_type == "value":
                value = self.agent.get_value(
                    states, temperature=self.log_temp.detach().exp(),
                    use_target=False, action_grad=False,
                )
            if self.use_target and self.next_v_use_target:
                with torch.no_grad():
                    next_value = self.agent.get_value(
                        next_states, temperature=self.log_temp.detach().exp(),
                        use_target=True, action_grad=False,
                    )
            else:
                # IQ's no-target objective differentiates V(s') through current Q.
                next_value = self.agent.get_value(
                    next_states, temperature=self.log_temp.detach().exp(),
                    use_target=False, action_grad=False,
                )
            qs = (self.agent.get_double_q_function(states, actions) if self.agent.double_critic
                  else (self.agent.get_q_function(states, actions),))
            losses = [self._iq_loss(q, value, next_value, dones, expert) for q in qs]
            critic_loss = torch.stack([item[0] for item in losses]).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            result.add_metric("critic/q_loss", critic_loss.item())
            result.add_metric("critic/q_value", torch.stack(qs).mean().item())
            for key in losses[0][1]: # Count the first q function loss
                result.add_metric("iq/" + key, torch.stack([item[1][key] for item in losses]).mean().item())

            if self.gradient_step % self.actor_update_interval == 0:
                for _ in range(self.num_actor_updates):
                    sampled_actions, log_prob, _ = self.agent.resample_action(states)
                    q = self.agent.get_q_function(states, sampled_actions).squeeze(-1)
                    actor_loss = (self.log_temp.detach().exp() * log_prob - q).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    if self.learn_temp:
                        coefficient = self.log_temp if self.update_log_temp else self.log_temp.exp()
                        temp_loss = -(coefficient * (log_prob.detach() + self.target_entropy)).mean()
                        self.temp_optimizer.zero_grad()
                        temp_loss.backward()
                        self.temp_optimizer.step()
                result.add_metric("actor/loss", actor_loss.item())
                result.add_metric("actor/entropy", -log_prob.mean().item())
                if self.learn_temp:
                    result.add_metric("temp/loss", temp_loss.item())
            result.add_metric("temp/value", self.log_temp.detach().exp().item())
            result.add_metric("iq/expert_fraction", float(expert_mask.mean()))
            if self.use_target and self.gradient_step % self.target_update_interval == 0:
                self.agent.update_target()
            self.gradient_step += 1
        return result
