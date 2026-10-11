"""Offline optimization of a fixed dataset, without collecting transitions."""

from abc import abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from .basealgorithm import BaseAlgorithm
from utils.evaluate import evaluate
from utils.utils import RunningMeanStd


class OfflineAlgorithm(BaseAlgorithm):
    """Reuse evaluation/checkpoints while replacing the online training loop."""

    def __init__(self, training_envs, testing_envs, buffer, agent, logger,
                 device, args, offline_args):
        super().__init__(training_envs, testing_envs, buffer, agent, logger,
                         device, False, args)
        self.batch_size = int(offline_args.batch_size)
        self.dataset_steps = buffer.buffer_size if buffer.full else buffer.pos
        if self.dataset_steps <= 0:
            raise ValueError("Offline training requires a nonempty dataset")
        if self.batch_size < 1 or self.update_step_per_epoch < 1 or self.total_epoch < 1:
            raise ValueError("Offline batch size, epochs and updates must be positive")
        self.dataset_size = self.dataset_steps * buffer.num_envs
        self.obs_rms = None
        self.logger.step_axis = "gradient"

    def initialize(self):
        """Fit frozen normalization statistics using only the offline dataset."""
        if self.args.env.obs_norm:
            self.obs_rms = RunningMeanStd()
            for observations in self.buffer.buffer["states"][:, :self.dataset_steps]:
                for start in range(0, len(observations), 4096):
                    self.obs_rms.update(observations[start:start + 4096])
        # Deliberately do not reset or step training_envs.

    def get_obs_rms(self):
        return self.obs_rms

    def interact_with_envs(self):
        raise RuntimeError("OfflineAlgorithm does not collect environment transitions")

    def _sample_batch(self):
        batch = self.buffer.sample(self.batch_size)
        if self.obs_rms is not None:
            for key in ("states", "next_states", "nstep_states"):
                if key in batch:
                    batch[key] = self.obs_rms.norm(batch[key]).astype(np.float32)
        return batch

    def update(self, batch=None, start_train=True):
        """Each update samples a new minibatch; the fixed buffer is never written."""
        for _ in range(self.update_step_per_epoch):
            result = self._update_policy(self._sample_batch())
            self.gradient_step += 1
        result.add_metric("offline/dataset_size", self.dataset_size)
        return result

    @abstractmethod
    def _update_policy(self, batch):
        """Optimize on one minibatch and return a Result."""

    def test_condition(self, epoch):
        return self.test_interval > 0 and self.test_episodes > 0 and epoch % self.test_interval == 0

    def save_condition(self, epoch):
        return self.save_interval > 0 and epoch % self.save_interval == 0

    def train_log_condition(self, epoch):
        return self.train_log_interval > 0 and epoch % self.train_log_interval == 0

    def save(self, pre_fix=""):
        self.agent.save(self.log_dir + "/" + pre_fix + "_model.pth")
        if self.obs_rms is not None:
            torch.save(self.obs_rms.state_dict(),
                       self.log_dir + "/" + pre_fix + "_obs_rms.pth")

    def _evaluate(self, epoch):
        # Evaluation is optional; its transitions never enter the dataset.
        result, score = evaluate(
            agent=self.agent, test_episodes=self.test_episodes,
            env_args=self.args.env, seed=self.seed + self.gradient_step,
            training_envs=self if self.obs_rms is not None else None,
            test_epsilon=self.test_epsilon,
        )
        self.logger.log_test(epoch, self.interaction_step, self.gradient_step, result)
        if score > self.best_score:
            self.best_score = score
            self.save(pre_fix="best")

    def run(self):
        try:
            self.initialize()
            for epoch in tqdm(range(self.total_epoch), desc="Epoch", unit="epoch"):
                self.current_epoch = epoch
                if self.test_condition(epoch):
                    self._evaluate(epoch)
                result = self.update()
                if self.train_log_condition(epoch):
                    self.logger.log_train(epoch, self.interaction_step,
                                          self.gradient_step, result)
                if self.save_condition(epoch):
                    self.save(pre_fix="newest")
            if self.test_interval > 0 and self.test_episodes > 0:
                self._evaluate(self.total_epoch)
            self.save(pre_fix="newest")
        finally:
            self.logger.close()
