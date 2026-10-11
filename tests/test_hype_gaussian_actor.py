"""Check HyPE's scaled and unscaled actor/data paths without environment rollout."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box
from torch.distributions import Independent, Normal

from run_hype import build_actor, validate_expert_actions
from utils.networks import DiagGaussianActor, MLPNetwork, TanhGaussianActor


class HyPEGaussianActorTest(unittest.TestCase):
    def setUp(self):
        self.obs = Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self.act = Box(-0.4, 0.4, (2,), dtype=np.float32)
        self.args = SimpleNamespace(
            rescale=False, state_dependent_std=True, hidden_sizes=[8, 8],
            activation="relu", initialize=True, common_net=False,
            action_bound_method="tanh",
        )

    def test_unscaled_actor_has_raw_gaussian_actions_and_log_prob(self):
        actor = build_actor(self.obs, self.act, self.args, "cpu")
        self.assertIsInstance(actor, DiagGaussianActor)
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            layers = [layer for layer in actor.mu.modules() if isinstance(layer, torch.nn.Linear)]
            layers[-1].bias.fill_(1.5)
        states = torch.ones(16, 3)
        mean, std = actor(states)
        actions, log_prob, u = actor.get_action(states, deterministic=True)
        torch.testing.assert_close(actions, torch.full((16, 2), 1.5))
        torch.testing.assert_close(log_prob, Independent(Normal(mean, std), 1).log_prob(actions))
        self.assertIsNone(u)
        actions, log_prob, _ = actor.get_action(states, deterministic=False)
        self.assertTrue(torch.any(actions.abs() > 0.4))
        torch.testing.assert_close(actor.get_log_prob(states, actions), log_prob)
        (actions.square().mean() + log_prob.mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in actor.parameters()))

    def test_scaled_actor_preserves_existing_initialization_and_density(self):
        self.args.rescale = True
        for common_net in (False, True):
            with self.subTest(common_net=common_net):
                self.args.common_net = common_net
                torch.manual_seed(7)
                actor = build_actor(self.obs, self.act, self.args, "cpu")
                torch.manual_seed(7)
                reference = TanhGaussianActor(
                    self.obs, self.act, MLPNetwork, device="cpu",
                    state_dependent_std=True, hidden_sizes=[8, 8],
                    activation=torch.nn.ReLU, initialize=True, common_net=common_net,
                )
                for name, parameter in actor.state_dict().items():
                    torch.testing.assert_close(parameter, reference.state_dict()[name], rtol=0, atol=0)
                actions, log_prob, u = actor.get_action(torch.ones(5, 3), deterministic=False)
                torch.testing.assert_close(actions, 0.4 * u.tanh())
                torch.testing.assert_close(actor.get_log_prob(torch.ones(5, 3), actions, u), log_prob)

    def test_expert_validation_accepts_raw_ppo_actions_only_in_unscaled_mode(self):
        actions = np.array([[-1.94, 2.07]], dtype=np.float32)
        original = actions.copy()
        validate_expert_actions(actions, self.act, rescale=False)
        np.testing.assert_array_equal(actions, original)
        with self.assertRaisesRegex(ValueError, "bounds"):
            validate_expert_actions(actions, self.act, rescale=True)
        for rescale in (False, True):
            validate_expert_actions(np.zeros((4, 2), dtype=np.float32), self.act, rescale)
            for bad in (np.nan, np.inf):
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_expert_actions(np.array([[bad, 0.0]]), self.act, rescale)

    def test_unscaled_shared_network_is_not_silently_ignored(self):
        self.args.common_net = True
        with self.assertRaisesRegex(ValueError, "common_net=false"):
            build_actor(self.obs, self.act, self.args, "cpu")


if __name__ == "__main__":
    unittest.main()
