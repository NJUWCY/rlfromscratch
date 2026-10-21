"""Deep PPIL: joint reward/Q optimization followed by SAC policy improvement.

Based on Viano et al., Proximal Point Imitation Learning, Appendix J/L.7,
and outputs/P2IL/deep_PPIL/iq.py (reward-maximization convention).
"""
import math

import numpy as np
import torch

from .sac import SAC
from utils.result import Result


class PPIL(SAC):
    def __init__(self, training_envs, testing_envs, buffer, expert_buffer,
                 agent, logger, device, args, rl_args):
        cfg = args.algorithm
        if args.env.absorbing or int(rl_args.nstep) != 1:
            raise ValueError("Deep PPIL requires absorbing=false and nstep=1")
        if buffer is expert_buffer or buffer.nstep != 1 or expert_buffer.nstep != 1:
            raise ValueError("Deep PPIL needs separate one-step online/expert buffers")
        if not expert_buffer.full and expert_buffer.pos == 0:
            raise ValueError("Deep PPIL requires nonempty expert data")
        if not hasattr(agent, "reward"):
            raise ValueError("Attach an independent reward network to the SACAgent")
        self.expert_buffer = expert_buffer
        self.inner_steps = int(cfg.inner_steps)
        self.actor_update_interval = int(cfg.actor_update_interval)
        self.num_actor_updates = int(cfg.num_actor_updates)
        self.tau = float(cfg.tau)
        self.regularize = bool(cfg.regularize)
        self.chi = bool(cfg.chi)
        self.r_reg_coef = float(cfg.r_reg_coef)
        self.q_reg_coef = float(cfg.q_reg_coef)
        self.weight_source = cfg.weight_source
        self.loss_type = cfg.loss_type
        if min(self.inner_steps, self.actor_update_interval, self.num_actor_updates,
               int(rl_args.batch_size), int(rl_args.target_update_interval)) < 1:
            raise ValueError("Update counts, intervals and batch size must be positive")
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("algorithm.tau must be finite and positive")
        for name in ("r_reg_coef", "q_reg_coef"):
            coefficient = getattr(self, name)
            if not math.isfinite(coefficient) or coefficient < 0:
                raise ValueError(f"algorithm.{name} must be finite and nonnegative")
        if self.weight_source not in {"expert", "online"}:
            raise ValueError("weight_source must be expert or online")
        if self.loss_type not in {"value_expert", "value", "v0"}:
            raise ValueError("loss_type must be value_expert, value or v0")
        super().__init__(training_envs, testing_envs, buffer, agent, logger, device, args, rl_args)
        # PPIL reference uses Adam and permits Hopper's beta1=0.
        for name, lr, parameters in (
            ("actor", self.actor_lr, self.agent.actor.parameters()),
            ("critic", self.critic_lr, self.agent.critic.parameters()),
            ("reward", float(cfg.reward_lr), self.agent.reward.parameters()),
        ):
            betas = tuple(float(v) for v in cfg[name + "_betas"])
            if len(betas) != 2 or not all(0 <= v < 1 for v in betas):
                raise ValueError(f"{name}_betas must contain two values in [0,1)")
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError(f"{name} learning rate must be finite and positive")
            setattr(self, name + "_optimizer", torch.optim.Adam(parameters, lr=lr, betas=betas))
        self.critic_updates = 0

    def _ppil_loss(self, q, reward, value, next_value, dones, expert, *, log_metrics=False):
        """Detached-softmax surrogate; identical first derivative to log-mean-exp."""
        bootstrap = self.gamma * (1 - dones) * next_value
        delta = reward + bootstrap - q
        mask = expert if self.weight_source == "expert" else ~expert
        residual = delta[mask].reshape(-1)
        weights = torch.softmax(self.tau * residual.detach(), dim=0)
        reward_loss = -reward[expert].mean() + (weights * reward[mask].reshape(-1)).sum()
        logistic_loss = (weights * (bootstrap - q)[mask].reshape(-1)).sum()
        if self.loss_type == "value_expert":
            value_loss = (value - bootstrap)[expert].mean()
        elif self.loss_type == "value":
            value_loss = (value - bootstrap).mean()
        else:
            # Reference v0 is a proxy using sampled expert states, not true s0.
            value_loss = (1 - self.gamma) * value[expert].mean()
        regularize_loss = q.new_zeros(())
        if self.regularize:
            if self.r_reg_coef > 0:
                regularize_loss = regularize_loss + self.r_reg_coef * reward.square().mean()
            if self.q_reg_coef > 0:
                regularize_loss = regularize_loss + self.q_reg_coef * q.square().mean()
                
        if self.chi and self.r_reg_coef > 0:
            regularize_loss = regularize_loss + self.r_reg_coef * reward[expert].square().mean()
        loss = reward_loss + logistic_loss + value_loss + regularize_loss
        if not log_metrics:
            return loss, {}
        with torch.no_grad():
            # The surrogate has the correct gradient but not the scalar LME value.
            lme = (torch.logsumexp(self.tau * residual, dim=0)
                   - math.log(residual.numel())) / self.tau
            objective = -reward[expert].mean() + lme + value_loss + regularize_loss
            return loss, dict(
                reward_loss=reward_loss.detach(), logistic_loss=logistic_loss.detach(),
                value_loss=value_loss.detach(), regularize_loss=regularize_loss.detach(),
                objective=objective.detach(), expert_reward=reward[expert].mean().detach(),
                policy_reward=reward[~expert].mean().detach(),
                weight_max=weights.max(), effective_samples=weights.square().sum().reciprocal(),
                q_value=q.mean().detach(),
            )


    def _update_policy(self):
        # B samples per source, fixed throughout all inner updates.
        batch = self.buffer.sample_mixed(2 * self.batch_size, self.expert_buffer, 0.5)
        mask = batch.pop("expert_mask")
        if self.args.env.obs_norm:
            for key in ("states", "next_states"):
                batch[key][mask] = self.training_envs._norm_obs(batch[key][mask])
        states, actions, next_states = [
            torch.as_tensor(batch[k], dtype=torch.float32, device=self.device)
            for k in ("states", "actions", "next_states")
        ]
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device).reshape(-1, 1)
        expert = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        self.agent.train()
        with Result("train") as result:
            for inner_step in range(self.inner_steps):
                if self.agent.double_critic:
                    q_values = self.agent.get_double_q_function(states, actions)
                else:
                    q_values = (self.agent.get_q_function(states, actions),)
                reward = self.agent.reward(states, actions)
                value = self.agent.get_value(states, temperature=self.log_temp.detach().exp(),
                                             use_target=False, action_grad=False)
                if self.use_target:
                    with torch.no_grad():
                        next_value = self.agent.get_value(
                            next_states, temperature=self.log_temp.detach().exp(),
                            use_target=True, action_grad=False)
                else:
                    # Essential: retain gradients through the current Q at s'.
                    next_value = self.agent.get_value(
                        next_states, temperature=self.log_temp.detach().exp(),
                        use_target=False, action_grad=False)
                # Each Q has its own residual, weights and regularization.
                # Share reward and soft values, then backpropagate the sum once.
                head_results = [
                    self._ppil_loss(
                        q, reward, value, next_value, dones, expert,
                        log_metrics=inner_step == self.inner_steps - 1 and index == 0,
                    )
                    for index, q in enumerate(q_values)
                ]
                loss = self.critic_loss_coef * sum(item[0] for item in head_results)
                self.critic_optimizer.zero_grad(set_to_none=True)
                self.reward_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.critic_optimizer.step()
                self.reward_optimizer.step()
                if self.use_target and self.critic_updates % self.target_update_interval == 0:
                    self.agent.update_target()
                self.critic_updates += 1

            # Report Q1 statistics from the final inner update, without scaling.
            metrics = head_results[0][1]

            if self.gradient_step % self.actor_update_interval == 0:
                for _ in range(self.num_actor_updates):
                    actions_pi, log_prob, _ = self.agent.resample_action(states)
                    q_pi = self.agent.get_q_function(states, actions_pi).squeeze(-1)
                    actor_loss = (self.log_temp.detach().exp() * log_prob - q_pi).mean()
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    if self.learn_temp:
                        coefficient = self.log_temp if self.update_log_temp else self.log_temp.exp()
                        temp_loss = -(coefficient * (log_prob.detach() + self.target_entropy)).mean()
                        self.temp_optimizer.zero_grad(set_to_none=True)
                        temp_loss.backward()
                        self.temp_optimizer.step()
                result.add_metric("actor/loss", actor_loss.detach())
                result.add_metric("actor/entropy", -log_prob.detach().mean())
            for name, value in metrics.items():
                result.add_metric("ppil/" + name, value)
            result.add_metric("critic/q_loss", loss.detach())
            result.add_metric("critic/q_value", metrics["q_value"])
            result.add_metric("ppil/critic_updates", self.critic_updates)
            result.add_metric("temp/value", self.log_temp.detach().exp())
        self.gradient_step += 1
        return result
