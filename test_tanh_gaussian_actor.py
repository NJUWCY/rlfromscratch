"""Regression checks for squashed Gaussian log-probs and their gradients."""
import unittest

import numpy as np
import torch
from gymnasium.spaces import Box
from torch.distributions import TanhTransform

from utils.networks import MLPNetwork, TanhGaussianActor


class TanhGaussianLogProbTests(unittest.TestCase):
    def make_actor(self, dtype=torch.float32, state_dependent_std=False):
        numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
        observations = Box(-np.inf, np.inf, (3,), dtype=numpy_dtype)
        actions = Box(
            np.array([-0.4, -1.5], dtype=numpy_dtype),
            np.array([0.4, 2.5], dtype=numpy_dtype),
            dtype=numpy_dtype,
        )
        return TanhGaussianActor(
            observations, actions, MLPNetwork,
            state_dependent_std=state_dependent_std,
            hidden_sizes=[], initialize=True, last_std=0.01,
        ).to(dtype=dtype)

    def test_sampling_and_recomputation_match_transformed_distribution(self):
        for dtype in (torch.float32, torch.float64):
            for state_dependent_std in (False, True):
                for deterministic in (False, True):
                    with self.subTest(dtype=dtype, state_dependent_std=state_dependent_std,
                                      deterministic=deterministic):
                        torch.manual_seed(7)
                        actor = self.make_actor(dtype, state_dependent_std)
                        states = torch.randn(8, 3, dtype=dtype)
                        actions, actual, u = actor.get_action(states, deterministic)
                        dist = actor.get_dist(states)
                        expected = (
                            dist.base_dist.log_prob(u)
                            - TanhTransform().log_abs_det_jacobian(u, u.tanh()).sum(-1)
                            - actor.scale.log().sum()
                        )
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(
                            actor.get_log_prob(states, actions, u=u), actual
                        )
                        torch.testing.assert_close(
                            actions, u.tanh() * actor.scale + actor.loc
                        )
                        self.assertEqual(actual.shape, (8,))
                        self.assertTrue(u.requires_grad)
                        self.assertTrue(torch.isfinite(actual).all())

    def test_saturated_latent_values_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                actor = self.make_actor(dtype)
                states = torch.zeros(2, 3, dtype=dtype)
                u = torch.tensor([[20.0, -20.0], [100.0, -100.0]],
                                 dtype=dtype, requires_grad=True)
                actions = u.tanh() * actor.scale + actor.loc
                actual = actor.get_log_prob(states, actions, u=u)
                expected = (
                    actor.get_dist(states).base_dist.log_prob(u)
                    - TanhTransform().log_abs_det_jacobian(u, u.tanh()).sum(-1)
                    - actor.scale.log().sum()
                )
                torch.testing.assert_close(actual, expected)
                gradient, = torch.autograd.grad(actual.sum(), u)
                # At zero states, orthogonal initialization gives mu=0, std=1.
                torch.testing.assert_close(gradient, -u + 2 * u.tanh())
                self.assertTrue(torch.isfinite(gradient).all())

    def test_saturated_sample_preserves_actor_entropy_gradient(self):
        actor = self.make_actor()
        with torch.no_grad():
            actor.mu.net[-1].bias.copy_(torch.tensor([20.0, -20.0]))
        states = torch.zeros(1, 3)
        _, log_probs, _ = actor.get_action(states, deterministic=True)
        gradient, = torch.autograd.grad(log_probs.sum(), actor.mu.net[-1].bias)
        torch.testing.assert_close(gradient, torch.tensor([2.0, -2.0]))

    def test_action_only_boundary_is_finite(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                actor = self.make_actor(dtype)
                states = torch.zeros(2, 3, dtype=dtype)
                actions = torch.stack([actor.low, actor.high]).requires_grad_()
                actual = actor.get_log_prob(states, actions)
                self.assertTrue(torch.isfinite(actual).all())
                gradient, = torch.autograd.grad(actual.sum(), actions)
                self.assertTrue(torch.isfinite(gradient).all())

    def test_action_only_interior_matches_distribution(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                actor = self.make_actor(dtype, state_dependent_std=True)
                states = torch.zeros(3, 3, dtype=dtype)
                normalized = torch.tensor([[-0.8, 0.7], [0.0, 0.0], [0.5, -0.6]],
                                          dtype=dtype)
                actions = normalized * actor.scale + actor.loc
                actual = actor.get_log_prob(states, actions)
                expected = actor.get_dist(states).log_prob(actions)
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
