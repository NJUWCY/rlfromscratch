"""Behavior cloning by maximum likelihood of expert actions."""

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

from .offline import OfflineAlgorithm
from utils import OPTIMIZER_DICT
from utils.networks import DiagGaussianActor, TanhGaussianActor, DiscreteProbabilityActor
from utils.result import Result


class BehaviorCloning(OfflineAlgorithm):
    """Minimize -E_dataset[log pi_actor(a | s)]; no critic or entropy objective."""

    def __init__(self, training_envs, testing_envs, buffer, agent, logger,
                 device, args, bc_args):
        actor = agent.actor
        if not isinstance(actor, (DiagGaussianActor, TanhGaussianActor,
                                  DiscreteProbabilityActor)):
            raise TypeError("BC requires a probabilistic actor: DiagGaussianActor, "
                            "TanhGaussianActor or DiscreteProbabilityActor")
        if isinstance(actor, DiscreteProbabilityActor):
            if not isinstance(agent.action_space, Discrete) or agent.action_space.start != 0:
                raise ValueError("DiscreteProbabilityActor requires zero-based Discrete actions")
        elif not isinstance(agent.action_space, Box):
            raise ValueError("Gaussian actors require a Box action space")
        if args.env.absorbing or getattr(buffer, "absorbing", False):
            raise ValueError("BC requires env.absorbing=false")
        super().__init__(training_envs, testing_envs, buffer, agent, logger,
                         device, args, bc_args)
        actions = buffer.buffer["actions"][:, :self.dataset_steps]
        if not np.isfinite(actions).all():
            raise ValueError("Expert actions must be finite")
        if isinstance(actor, DiscreteProbabilityActor):
            if np.any(actions != np.floor(actions)) or np.any(actions < 0) or np.any(actions >= self.action_space.n):
                raise ValueError("Expert actions must be valid integer action indices")
        elif isinstance(actor, TanhGaussianActor):
            if (np.any(actions < self.action_space.low - 1e-5)
                    or np.any(actions > self.action_space.high + 1e-5)):
                raise ValueError("TanhGaussianActor requires expert actions within environment bounds")
        # Gaussian std parameters and a discrete policy's encoder belong to actor.
        self.actor_optimizer = OPTIMIZER_DICT[bc_args.actor_optimizer](
            self.agent.actor.parameters(), lr=bc_args.actor_lr)
        self.l2_coef = float(getattr(bc_args, "l2_coef", 0.0))
        if not np.isfinite(self.l2_coef) or self.l2_coef < 0:
            raise ValueError("l2_coef must be finite and non-negative")

    def _update_policy(self, batch):
        self.agent.actor.train()
        with Result("train") as result:
            log_probs = self.agent.log_prob(batch["states"], batch["actions"])
            if log_probs.ndim != 1 or len(log_probs) != len(batch["states"]):
                raise ValueError("The policy must return one joint log-probability per action")
            nll = -log_probs.mean()
            if not torch.isfinite(nll):
                raise FloatingPointError("BC negative log-likelihood is not finite")
            # Explicit L2 penalty on every actor parameter, l2_coef * sum ||theta||^2.
            l2_norm = sum(p.pow(2).sum() for p in self.agent.actor.parameters() if p.requires_grad)
            loss = nll + self.l2_coef * l2_norm if self.l2_coef > 0 else nll
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
        result.add_metric("actor/loss", loss.item())
        result.add_metric("actor/nll", nll.item())
        result.add_metric("actor/l2_norm", l2_norm.item())
        result.add_metric("actor/log_prob", log_probs.detach().mean().item())
        result.add_metric("actor/lr", self.actor_optimizer.param_groups[0]["lr"])
        return result
