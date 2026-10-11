"""SQIL for continuous control, using the library's SAC implementation."""

import numpy as np

from .sac import SAC


class SQIL(SAC):
    """Learn from scheduled mixtures of expert (+1) and online (0) transitions.

    Demonstrations stay in a separate ExpertDataset and are never overwritten
    by online interaction. The environment's rewards are used only for logging
    and evaluation. This is the SAC variant of SQIL (Reddy et al., Section 4.3),
    with equal sampling by default (lambda_samp=1). An optional HyPE-style
    schedule changes the expert fraction as SAC updates accumulate.
    """

    def __init__(self, training_envs, testing_envs, buffer, expert_buffer,
                 agent, logger, device, args, rl_args):
        if int(rl_args.batch_size) < 1:
            raise ValueError("SQIL requires batch_size >= 1")
        if int(rl_args.nstep) != 1 or buffer.nstep != 1 or expert_buffer.nstep != 1:
            raise ValueError("SQIL currently supports nstep=1")
        if args.env.absorbing or getattr(expert_buffer, "absorbing", False):
            raise ValueError("SQIL requires env.absorbing=false; do not add DAC absorbing self-loops")
        if buffer is expert_buffer:
            raise ValueError("SQIL needs separate online and expert buffers")
        if expert_buffer.buffer_size <= 0 or (not expert_buffer.full and expert_buffer.pos == 0):
            raise ValueError("SQIL requires a nonempty expert dataset")
        algorithm_args = getattr(args, "algorithm", None)
        self.sampling_schedule = np.asarray(
            getattr(algorithm_args, "sampling_schedule", [[0.5, 0.5, 1000000]]),
            dtype=np.float64,
        )
        schedule = self.sampling_schedule
        if (schedule.ndim != 2 or schedule.shape[1] != 3 or len(schedule) == 0
                or not np.isfinite(schedule).all()
                or np.any(schedule[:, :2] < 0) or np.any(schedule[:, :2] > 1)
                or np.any(np.diff(np.r_[0, schedule[:, 2]]) <= 0)):
            raise ValueError("Expected [start_ratio, end_ratio, increasing_end_step] rows")
        super().__init__(training_envs, testing_envs, buffer, agent, logger,
                         device, args, rl_args)
        self.expert_buffer = expert_buffer

    def _update_buffer(self, batch):
        # Preserve the caller's true rewards and the episode-return logs.
        online_batch = dict(batch)
        online_batch["rewards"] = np.zeros_like(batch["rewards"], dtype=np.float32)
        self.buffer.add(online_batch)

    def _expert_sampling_ratio(self, step=None):
        """Use HyPE's piecewise-linear schedule of completed SAC updates."""
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
        expert_mask = batch.pop("expert_mask")

        # Match the library's collection-time normalization of online states.
        # Expert observations remain raw in the dataset and are normalized only
        # in this sampled batch, without updating the running statistics.
        if self.args.env.obs_norm and expert_mask.any():
            for key in ("states", "next_states"):
                batch[key][expert_mask] = self.training_envs._norm_obs(batch[key][expert_mask])

        # Label by sample source, never by the original task reward or by an
        # expert-state/action matching test. Even terminal expert actions get 1;
        # SAC's done mask makes their future value zero. Timeouts bootstrap.
        batch["rewards"] = expert_mask.astype(np.float32)
        result = self._sac_update(batch)
        result.add_metric("sqil/expert_ratio", expert_ratio)
        result.add_metric("sqil/expert_fraction", float(expert_mask.mean()))
        result.add_metric("sqil/sample_step", self.gradient_step)
        result.add_metric("sqil/reward_mean", float(batch["rewards"].mean()))
        return result
