"""CPU checks for the optional garage network/loss settings; no environment rollout."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.sac.policies import Actor as SB3Actor

from agent import SACAgent
from algorithm.sac import SAC
from run_hype import build_discriminator_network
from utils.networks import DoubleQFunction, MLPNetwork, TanhGaussianActor

# Compare against the actual reference network kept alongside this project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs" / "garage"))
try:
    from garage.models.discriminator import Discriminator as GarageDiscriminator
finally:
    sys.path.pop(0)


class GarageNetworkTest(unittest.TestCase):
    def setUp(self):
        self.obs = Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self.act = Box(-0.4, 0.4, (2,), dtype=np.float32)

    def actor(self, **kwargs):
        options = dict(hidden_sizes=[8, 8], activation=torch.nn.ReLU,
                       state_dependent_std=True, initialize=False)
        options.update(kwargs)
        return TanhGaussianActor(self.obs, self.act, MLPNetwork, **options)

    def disc_args(self, mode="orthogonal", spectral_norm=False):
        return SimpleNamespace(
            discriminator_initialization=mode,
            discriminator_hidden_sizes=[256, 256],
            discriminator_activation="relu", discriminator_output_gain=1.0,
            spectral_norm=spectral_norm,
        )

    def assert_parameters_equal(self, left, right):
        lp, rp = list(left.parameters()), list(right.parameters())
        self.assertEqual(len(lp), len(rp))
        for a, b in zip(lp, rp):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_shared_actor_matches_sb3_initialization_forward_and_gradients(self):
        torch.manual_seed(23)
        actor = self.actor(common_net=True)
        torch.manual_seed(23)
        reference = SB3Actor(self.obs, self.act, [8, 8],
                             FlattenExtractor(self.obs), 3, activation_fn=torch.nn.ReLU)
        self.assert_parameters_equal(actor, reference)
        states = torch.randn(7, 3)
        mu, std = actor(states)
        ref_mu, ref_log_std, _ = reference.get_action_dist_params(states)
        torch.testing.assert_close(mu, ref_mu)
        torch.testing.assert_close(std, ref_log_std.exp())
        grad = torch.autograd.grad((mu.square() + std.square()).mean(), actor.parameters())
        ref_grad = torch.autograd.grad(
            (ref_mu.square() + ref_log_std.exp().square()).mean(), reference.parameters())
        for a, b in zip(grad, ref_grad):
            torch.testing.assert_close(a, b)

    def test_actor_variants_preserve_scaled_actions_log_prob_and_old_checkpoints(self):
        for hidden_sizes in ([], [8, 8]):
            for state_dependent_std in (False, True):
                for common_net in (False, True):
                    with self.subTest(hidden=hidden_sizes, std=state_dependent_std, shared=common_net):
                        actor = self.actor(hidden_sizes=hidden_sizes,
                                           state_dependent_std=state_dependent_std,
                                           common_net=common_net)
                        states = torch.randn(5, 3)
                        actions, log_prob, u = actor.get_action(states, deterministic=False)
                        torch.testing.assert_close(actions, 0.4 * u.tanh())
                        torch.testing.assert_close(actor.get_log_prob(states, actions, u), log_prob)
                        self.assertTrue(torch.isfinite(log_prob).all())
                        log_prob.mean().backward()
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                            for p in actor.parameters()))
                        restored = self.actor(hidden_sizes=hidden_sizes,
                                              state_dependent_std=state_dependent_std,
                                              common_net=common_net)
                        restored.load_state_dict(actor.state_dict())
                        if not common_net:
                            del restored.common_net  # Older pickled actors lack this field.
                            self.assertTrue(all(k.startswith(("mu.", "log_sigma"))
                                                for k in restored.state_dict()))
                        for a, b in zip(actor(states), restored(states)):
                            torch.testing.assert_close(a, b)

    def test_discriminator_matches_garage_weights_biases_and_gradients(self):
        env = SimpleNamespace(observation_space=self.obs, action_space=self.act)
        torch.manual_seed(19)
        reference = GarageDiscriminator(env)
        torch.manual_seed(19)
        network = build_discriminator_network(5, self.disc_args())
        self.assert_parameters_equal(network, reference)
        states_actions = torch.randn(6, 5)
        actual, expected = network(states_actions).squeeze(-1), reference(states_actions)
        torch.testing.assert_close(actual, expected)
        grad = torch.autograd.grad(actual.square().mean(), network.parameters())
        ref_grad = torch.autograd.grad(expected.square().mean(), reference.parameters())
        for a, b in zip(grad, ref_grad):
            torch.testing.assert_close(a, b)

    def test_discriminator_modes_and_spectral_norm(self):
        for mode in ("orthogonal_relu", "default"):
            torch.manual_seed(11)
            actual = build_discriminator_network(5, self.disc_args(mode))
            torch.manual_seed(11)
            expected = MLPNetwork(5, 1, [256, 256], torch.nn.ReLU,
                                  initialize=mode == "orthogonal_relu", last_std=1.0)
            self.assert_parameters_equal(actual, expected)
        actual = build_discriminator_network(5, self.disc_args(spectral_norm=True))
        layers = [layer for layer in actual.modules() if isinstance(layer, torch.nn.Linear)]
        for layer in layers:
            weight = layer.parametrizations.weight.original.detach()
            gram = weight.T @ weight if weight.shape[0] >= weight.shape[1] else weight @ weight.T
            torch.testing.assert_close(gram, torch.eye(len(gram)), rtol=1e-5, atol=1e-5)
        actual(torch.randn(4, 5)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in actual.parameters()))
        for mode in ("unknown", "inherit"):
            with self.assertRaises(ValueError):
                build_discriminator_network(5, self.disc_args(mode))

    def test_critic_coefficient_scales_gradients_in_real_sac_update(self):
        # Inspect gradients at critic.step(), before actor.backward() also touches Q.
        class RecordingSGD(torch.optim.SGD):
            def step(self, closure=None):
                self.recorded = [p.grad.detach().clone()
                                 for group in self.param_groups for p in group["params"]]
                return super().step(closure)

        rng = np.random.default_rng(7)
        batch = dict(states=rng.normal(size=(4, 3)).astype(np.float32),
                     actions=rng.uniform(-0.4, 0.4, size=(4, 2)).astype(np.float32),
                     next_states=rng.normal(size=(4, 3)).astype(np.float32),
                     rewards=rng.normal(size=4).astype(np.float32),
                     dones=np.ones(4, dtype=np.float32),
                     nstep_gamma=np.full(4, 0.98, dtype=np.float32))
        recorded = []
        for coef in (1.0, 0.5):
            torch.manual_seed(31)
            actor = self.actor(common_net=True)
            critic = DoubleQFunction(self.obs, self.act, MLPNetwork,
                                     hidden_sizes=[8, 8], activation=torch.nn.ReLU)
            algorithm = SAC.__new__(SAC)
            algorithm.agent = SACAgent(self.obs, self.act, "cpu", actor, critic)
            algorithm.device = torch.device("cpu")
            algorithm.critic_loss_coef = coef
            algorithm.log_temp = torch.nn.Parameter(torch.zeros(()))
            algorithm.learn_temp = False
            algorithm.use_target = False
            algorithm.gradient_step = 0
            algorithm.critic_optimizer = RecordingSGD(critic.parameters(), lr=0.0)
            algorithm.actor_optimizer = torch.optim.SGD(actor.parameters(), lr=0.0)
            qs = algorithm.agent.get_double_q_function(batch["states"], batch["actions"])
            target = torch.from_numpy(batch["rewards"]).unsqueeze(1)
            expected = torch.autograd.grad(
                coef * sum(torch.nn.functional.mse_loss(q, target) for q in qs),
                critic.parameters())
            algorithm._sac_update(batch)
            actual = algorithm.critic_optimizer.recorded
            self.assertEqual(algorithm.gradient_step, 1)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
            recorded.append(actual)
        for original, scaled in zip(*recorded):
            torch.testing.assert_close(scaled, 0.5 * original)


if __name__ == "__main__":
    unittest.main()
