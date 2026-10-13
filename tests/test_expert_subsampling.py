"""Shared expert subsampling: offsets, paired fields and reproducibility."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from gymnasium.spaces import Box

from memory import ExpertDataset


class ExpertSubsamplingTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'expert.npz'
        ids = np.arange(15, dtype=np.float32)
        self.source = dict(
            observations=ids[:, None], next_observations=(ids + 0.5)[:, None],
            actions=(ids / 20)[:, None], rewards=ids + 100,
            terminals=np.isin(ids, [7]), timeouts=np.isin(ids, [13, 14]),
            episode_ends=np.isin(ids, [7, 13, 14]),
        )
        np.savez(self.path, **self.source)
        self.obs = Box(-np.inf, np.inf, (1,), dtype=np.float32)
        self.act = Box(-1, 1, (1,), dtype=np.float32)

    def load(self, frequency=3, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return ExpertDataset(str(self.path), self.obs, self.act, 3, frequency, **kwargs)

    def assert_paired_fields(self, dataset, indices):
        mapping = dict(states='observations', next_states='next_observations',
                       actions='actions', rewards='rewards', dones='terminals',
                       truncateds='timeouts')
        for output, source in mapping.items():
            np.testing.assert_array_equal(dataset.buffer[output][0], self.source[source][indices])

    def test_independent_offsets_preserve_transitions_and_short_episode(self):
        # Episodes have lengths 8, 6 and 1. Deliberately choose different offsets.
        with patch('numpy.random.choice', return_value=np.array([0, 1, 2])), \
                patch('numpy.random.randint', side_effect=[1, 2, 0]) as choose_start:
            dataset = self.load()
        self.assertEqual([call.args[0] for call in choose_start.call_args_list], [3, 3, 1])
        self.assert_paired_fields(dataset, [1, 4, 7, 10, 13, 14])
        np.testing.assert_array_equal(dataset.episode_ends, [2, 4, 5])

    def test_fixed_start_and_frequency_one_do_not_draw_offsets(self):
        for frequency, kwargs, indices in (
            (3, dict(random_start=False), [0, 3, 6, 8, 11, 14]),
            (1, {}, list(range(15))),
        ):
            with self.subTest(frequency=frequency), \
                    patch('numpy.random.choice', return_value=np.array([0, 1, 2])), \
                    patch('numpy.random.randint', side_effect=AssertionError('unexpected offset draw')):
                self.assert_paired_fields(self.load(frequency, **kwargs), indices)

    def test_seed_reproduces_selected_episodes_and_offsets(self):
        np.random.seed(123)
        first = self.load()
        np.random.seed(123)
        second = self.load()
        self.assertEqual(first.subsample_starts, second.subsample_starts)
        for key in first.buffer:
            np.testing.assert_array_equal(first.buffer[key], second.buffer[key])


if __name__ == '__main__':
    unittest.main()
