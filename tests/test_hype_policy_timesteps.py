"""Policy-clock regression checks without MuJoCo or GPU training."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from algorithm.hype import HyPESAC
from utils.result import Result


class PolicyTimestepsTest(unittest.TestCase):
    def make_algorithm(self, reset, *, warmup=64, interval=4, epochs=8,
                       num_envs=1, decay=True):
        algorithm = HyPESAC.__new__(HyPESAC)
        algorithm.reset_policy_timesteps = reset
        algorithm.policy_lr_decay = decay
        algorithm.interaction_step = 0
        algorithm.policy_interaction_step = 0
        algorithm._policy_phase_start_step = 0
        algorithm.gradient_step = 0
        algorithm.start_train_step = warmup
        algorithm.discriminator_train_interval = interval
        algorithm.interact_per_epoch = 64
        algorithm.num_training_envs = num_envs
        algorithm.total_epoch = epochs
        algorithm.update_step_per_epoch = 1
        algorithm.update_discriminator_step_per_epoch = 1
        algorithm.discriminator_round = 0
        algorithm.disc_lr = 1e-3
        algorithm.discriminator_args = SimpleNamespace(
            discriminator_optimizer="Adam", discriminator_weight_decay=0.0)
        algorithm.discriminator = torch.nn.Linear(1, 1)
        algorithm.training_envs = Mock()
        algorithm.batch_size = 4
        algorithm.expert_buffer = object()
        algorithm.sampling_schedule = np.array([[0.2, 0.0, 1000.0]])
        algorithm.learn_temp = True
        algorithm.actor_lr, algorithm.critic_lr, algorithm.temp_lr = 0.1, 0.2, 0.3
        optimizers = []
        for name in ("actor", "critic", "temp"):
            parameter = torch.nn.Parameter(torch.tensor(1.0))
            optimizer = torch.optim.Adam([parameter], lr=getattr(algorithm, name + "_lr"))
            setattr(algorithm, name + "_optimizer", optimizer)
            optimizers.append(optimizer)

        recorded = SimpleNamespace(ratios=[], lr=[], batches=[], random_steps=0)

        def sample_mixed(batch_size, expert_buffer, ratio):
            recorded.ratios.append(ratio)
            return {"expert_mask": np.zeros(batch_size, dtype=bool),
                    "rewards": np.zeros(batch_size, dtype=np.float32)}

        algorithm.buffer = SimpleNamespace(sample_mixed=sample_mixed)

        def add_buffer(batch):
            recorded.batches.append(batch)
            return Result("buffer")

        algorithm._update_buffer = add_buffer
        algorithm._nstep_discriminator_rewards = lambda batch: batch["rewards"]

        def sac_update(batch):
            recorded.lr.append(tuple(o.param_groups[0]["lr"] for o in optimizers))
            for optimizer in optimizers:
                parameter = optimizer.param_groups[0]["params"][0]
                optimizer.zero_grad()
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
            algorithm.gradient_step += 1
            return Result("train")

        algorithm._sac_update = sac_update

        def collect_discriminator_samples():
            # Extra D interactions must not count towards the next warmup.
            algorithm.interaction_step += 11
            return 11

        algorithm._collect_discriminator_samples = collect_discriminator_samples
        algorithm._update_discriminator = lambda: Result("discriminator")
        return algorithm, recorded, optimizers

    def run_epochs(self, algorithm, recorded):
        for epoch in range(algorithm.total_epoch):
            for _ in range(algorithm.interact_per_epoch):
                if algorithm.random_choose_action():
                    recorded.random_steps += algorithm.num_training_envs
                algorithm.interaction_step += algorithm.num_training_envs
            algorithm.current_epoch = epoch
            batch = {"states": np.zeros(
                (algorithm.num_training_envs, algorithm.interact_per_epoch, 1))}
            algorithm.update(batch, algorithm.start_train())

    def test_continuous_mode_preserves_existing_clock(self):
        algorithm, recorded, _ = self.make_algorithm(False)
        self.run_epochs(algorithm, recorded)
        self.assertEqual(recorded.random_steps, 64)
        self.assertEqual(algorithm.gradient_step, 8)  # Existing >= warmup gate.
        self.assertEqual(algorithm.discriminator_round, 2)
        np.testing.assert_allclose([lr[0] for lr in recorded.lr],
                                   0.1 * (1 - np.arange(1, 9) / 8))
        self.assertEqual(algorithm.policy_interaction_step, 512)

    def test_reset_repeats_warmup_and_lr_but_preserves_training_state(self):
        algorithm, recorded, optimizers = self.make_algorithm(True)
        buffer = algorithm.buffer
        self.run_epochs(algorithm, recorded)
        self.assertEqual(recorded.random_steps, 128)
        self.assertEqual(algorithm.gradient_step, 6)  # SB3's strict > warmup gate.
        self.assertEqual(algorithm.discriminator_round, 2)
        self.assertEqual(algorithm.policy_interaction_step, 512)
        self.assertEqual(algorithm.interaction_step, 534)
        self.assertEqual(algorithm._policy_phase_step(), 0)
        self.assertEqual(len(recorded.batches), 8)
        self.assertIs(algorithm.buffer, buffer)
        for name, optimizer in zip(("actor", "critic", "temp"), optimizers):
            self.assertIs(getattr(algorithm, name + "_optimizer"), optimizer)
            parameter = optimizer.param_groups[0]["params"][0]
            self.assertEqual(optimizer.state[parameter]["step"].item(), 6)
        np.testing.assert_allclose(recorded.lr,
            [[0.05, 0.1, 0.15], [0.025, 0.05, 0.075], [0, 0, 0]] * 2)
        np.testing.assert_allclose(recorded.ratios, 0.2 * (1 - np.arange(6) / 1000))
        algorithm.training_envs.reset.assert_not_called()

    def test_phase_can_finish_entirely_inside_warmup(self):
        algorithm, recorded, _ = self.make_algorithm(True, warmup=512)
        self.run_epochs(algorithm, recorded)
        self.assertEqual(recorded.random_steps, 512)
        self.assertEqual(algorithm.gradient_step, 0)
        self.assertEqual(algorithm.discriminator_round, 2)
        self.assertEqual(algorithm._policy_phase_step(), 0)
        self.assertEqual(len(recorded.batches), 8)

    def test_disabling_lr_decay_keeps_all_learning_rates_constant(self):
        algorithm, recorded, _ = self.make_algorithm(True, decay=False)
        self.run_epochs(algorithm, recorded)
        np.testing.assert_allclose(recorded.lr, [[0.1, 0.2, 0.3]] * 6)

    def test_vector_steps_are_counted_as_transitions(self):
        algorithm, recorded, _ = self.make_algorithm(True, warmup=192, num_envs=3)
        self.run_epochs(algorithm, recorded)
        self.assertEqual(recorded.random_steps, 384)
        self.assertEqual(algorithm.policy_interaction_step, 1536)
        self.assertEqual(algorithm.gradient_step, 6)
        np.testing.assert_allclose([lr[0] for lr in recorded.lr],
                                   [0.05, 0.025, 0] * 2)

    def test_default_halfcheetah_phase_boundary(self):
        algorithm, recorded, _ = self.make_algorithm(
            True, warmup=1000, interval=157, epochs=314)
        self.run_epochs(algorithm, recorded)
        self.assertEqual(recorded.random_steps, 2000)
        self.assertEqual(algorithm.policy_interaction_step, 20096)
        self.assertEqual(algorithm.gradient_step, 284)
        self.assertEqual(algorithm.discriminator_round, 2)
        np.testing.assert_allclose(recorded.lr[:142], recorded.lr[142:])
        self.assertEqual(recorded.lr[141], (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
