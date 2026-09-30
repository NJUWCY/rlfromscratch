"""DAC reward inference must not update spectral-normalization buffers."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box

from algorithm.dac import DAC
from discriminator.discriminator import GAILDiscriminator
from utils.networks import MLPNetwork


class FixedBuffer:
    def __init__(self, batch):
        self.batch = batch

    def sample(self, batch_size):
        return {key: value[:batch_size].copy() for key, value in self.batch.items()}


class DACDiscriminatorModeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        rng = np.random.default_rng(5)
        obs_space = Box(-np.inf, np.inf, (4,), dtype=np.float32)
        action_space = Box(-0.4, 0.4, (2,), dtype=np.float32)
        net = MLPNetwork(6, 1, hidden_sizes=[8], activation=torch.nn.ReLU,
                         initialize=True, spectral_norm=True)
        self.disc = GAILDiscriminator(
            obs_space, action_space, torch.device("cpu"), net, reward_function="AIRL"
        )
        self.dac = DAC.__new__(DAC)
        self.dac.discriminator = self.disc
        self.dac.device = torch.device("cpu")
        self.dac.gamma = 0.97
        self.dac.args = SimpleNamespace(env=SimpleNamespace(obs_norm=False))
        self.dac.discriminator_train_steps = 1
        self.dac.discriminator_batch_size = 16
        self.dac.discriminator_gradient_penalty = True
        self.dac.gradient_penalty_coef = 1.0
        self.dac.target_gradient_norm = 1.0
        self.dac.discriminator_optimizer = torch.optim.AdamW(
            self.disc.parameters(), lr=3e-5, weight_decay=10.0
        )
        self.dac.buffer = FixedBuffer({
            "states": rng.normal(size=(16, 4)).astype(np.float32),
            "actions": rng.uniform(-0.4, 0.4, size=(16, 2)).astype(np.float32),
        })
        self.dac.expert_buffer = FixedBuffer({
            "states": rng.normal(1.0, size=(16, 4)).astype(np.float32),
            "actions": rng.uniform(-0.4, 0.4, size=(16, 2)).astype(np.float32),
        })
        self.batch = {
            "nstep_states": rng.normal(size=(16, 3, 4)).astype(np.float32),
            "nstep_actions": rng.uniform(-0.4, 0.4, size=(16, 3, 2)).astype(np.float32),
            "nstep_mask": np.ones((16, 3), dtype=np.float32),
        }
        self.batch["nstep_mask"][::2, -1] = 0.0
        # A real optimizer update leaves the spectral estimates ready to change
        # on the next training-mode forward, reproducing the reward drift.
        self.dac._update_discriminator()

    def test_reward_is_repeatable_and_preserves_state_and_mode(self):
        for training in (True, False):
            with self.subTest(training=training):
                self.disc.train(training)
                before = {name: value.clone() for name, value in self.disc.state_dict().items()}
                forward_modes = []
                hook = self.disc.network.register_forward_pre_hook(
                    lambda module, inputs: forward_modes.append(
                        (module.training, torch.is_grad_enabled())
                    )
                )
                try:
                    first = self.dac._nstep_discriminator_rewards(self.batch)
                    second = self.dac._nstep_discriminator_rewards(self.batch)
                finally:
                    hook.remove()
                np.testing.assert_array_equal(first, second)
                self.assertTrue(np.isfinite(first).all())
                self.assertEqual(forward_modes, [(False, False), (False, False)])
                self.assertTrue(all(module.training == training for module in self.disc.modules()))
                for name, value in self.disc.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_reward_exception_restores_mode(self):
        for training in (True, False):
            with self.subTest(training=training):
                self.disc.train(training)

                def fail(*args, **kwargs):
                    self.assertFalse(self.disc.training)
                    self.assertFalse(torch.is_grad_enabled())
                    raise RuntimeError("injected reward failure")

                with patch.object(self.disc, "predict_reward", side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, "injected reward failure"):
                        self.dac._nstep_discriminator_rewards(self.batch)
                self.assertTrue(all(module.training == training for module in self.disc.modules()))

    def test_training_resumes_and_updates_parameters_and_spectral_buffers(self):
        self.disc.eval()
        reward_before = self.dac._nstep_discriminator_rewards(self.batch)
        params_before = {name: value.clone() for name, value in self.disc.named_parameters()}
        buffers_before = {name: value.clone() for name, value in self.disc.named_buffers()}
        self.dac._update_discriminator()
        self.assertTrue(all(module.training for module in self.disc.modules()))
        self.assertTrue(any(
            not torch.equal(value, params_before[name])
            for name, value in self.disc.named_parameters()
        ))
        self.assertTrue(any(
            not torch.equal(value, buffers_before[name])
            for name, value in self.disc.named_buffers()
        ))
        for parameter in self.disc.parameters():
            self.assertTrue(torch.isfinite(parameter).all())
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        reward_after = self.dac._nstep_discriminator_rewards(self.batch)
        self.assertFalse(np.array_equal(reward_before, reward_after))


if __name__ == "__main__":
    unittest.main()
