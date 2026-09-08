import unittest

from src.data.source import InMemoryTrajectorySource
from src.data.temporal_dataset import TemporalSequenceDataset
from src.data.trajectory import Trajectory


def make_trajectory(num_steps: int, *, terminal: bool = True) -> Trajectory:
    observations = [f"s{i}" for i in range(num_steps + 1)]
    actions = [f"a{i}" for i in range(num_steps)]
    rewards = [float(i) for i in range(num_steps)]
    terminated = [False] * num_steps
    truncated = [False] * num_steps
    infos = [{} for _ in range(num_steps)]

    if terminal and num_steps:
        terminated[-1] = True

    return Trajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        infos=infos,
    )


class TrajectoryTests(unittest.TestCase):
    def test_observation_action_invariant(self):
        trajectory = make_trajectory(4)
        self.assertEqual(len(trajectory.observations), 5)
        self.assertEqual(len(trajectory.actions), 4)
        self.assertTrue(trajectory.done)


class TemporalDatasetTests(unittest.TestCase):
    def test_frame_skip_is_applied_after_collection(self):
        source = InMemoryTrajectorySource([make_trajectory(8)])
        dataset = TemporalSequenceDataset(
            source,
            history_size=2,
            frame_skip=2,
        )

        sample = dataset[0]
        self.assertEqual(sample.observations, ["s0", "s2", "s4"])
        self.assertEqual(sample.action_chunks, [["a0", "a1"], ["a2", "a3"]])
        self.assertEqual(sample.reward_chunks, [[0.0, 1.0], [2.0, 3.0]])

    def test_partial_terminal_chunk_is_preserved_when_enabled(self):
        source = InMemoryTrajectorySource([make_trajectory(5, terminal=True)])
        dataset = TemporalSequenceDataset(
            source,
            history_size=2,
            frame_skip=3,
            allow_partial_final_chunk=True,
        )

        sample = dataset[0]
        self.assertEqual(sample.observations, ["s0", "s3", "s5"])
        self.assertEqual(sample.action_chunks, [["a0", "a1", "a2"], ["a3", "a4"]])
        self.assertEqual(sample.chunk_lengths, [3, 2])
        self.assertTrue(sample.terminated[-1])

    def test_partial_terminal_chunk_is_not_exposed_to_fixed_width_model_when_disabled(self):
        source = InMemoryTrajectorySource([make_trajectory(5, terminal=True)])
        dataset = TemporalSequenceDataset(
            source,
            history_size=2,
            frame_skip=3,
            allow_partial_final_chunk=False,
        )
        self.assertEqual(len(dataset), 0)


if __name__ == "__main__":
    unittest.main()
