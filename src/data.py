from copy import deepcopy
from dataclasses import dataclass, field
import random
from typing import Any, Callable

from torch.utils.data import Dataset

from src.interfaces.environment import EnvironmentInterface


@dataclass
class Trajectory:
    """Raw environment trajectory. No frame skipping or model preprocessing."""

    observations: list[Any]
    actions: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    @property
    def done(self) -> bool:
        return self.num_steps > 0 and (self.terminated[-1] or self.truncated[-1])

    def append_step(
        self,
        *,
        action: Any,
        reward: float,
        next_observation: Any,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.infos.append(info)
        self.observations.append(next_observation)

    def validate(self) -> None:
        n = len(self.actions)
        if len(self.observations) != n + 1:
            raise ValueError(
                "Trajectory invariant violated: expected "
                "len(observations) == len(actions) + 1"
            )

        for name, values in {
            "rewards": self.rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "infos": self.infos,
        }.items():
            if len(values) != n:
                raise ValueError(
                    f"Trajectory invariant violated: len({name})={len(values)}, "
                    f"len(actions)={n}"
                )


ActionPolicy = Callable[[EnvironmentInterface[Any, Any], Any], Any]


def collect_trajectories(
    env: EnvironmentInterface[Any, Any],
    *,
    num_env_steps: int,
    seed: int | None = None,
    policy: ActionPolicy | None = None,
) -> list[Trajectory]:
    """Collect every raw simulator transition; frame_skip is applied later."""

    if num_env_steps < 1:
        raise ValueError("num_env_steps must be >= 1")

    policy = policy or (lambda current_env, observation: current_env.sample_action())
    trajectories: list[Trajectory] = []

    observation, reset_info = env.reset(seed=seed)
    episode_id = 0
    trajectory = Trajectory(
        observations=[deepcopy(observation)],
        metadata={
            "episode_id": episode_id,
            "reset_info": deepcopy(reset_info),
            "collection_cutoff": False,
        },
    )

    for global_step in range(num_env_steps):
        action = policy(env, observation)
        result = env.step(action)

        trajectory.append_step(
            action=deepcopy(action),
            reward=result.reward,
            next_observation=deepcopy(result.observation),
            terminated=result.terminated,
            truncated=result.truncated,
            info=deepcopy(result.info),
        )
        observation = result.observation

        if result.done:
            trajectories.append(trajectory)
            episode_id += 1

            if global_step + 1 < num_env_steps:
                observation, reset_info = env.reset()
                trajectory = Trajectory(
                    observations=[deepcopy(observation)],
                    metadata={
                        "episode_id": episode_id,
                        "reset_info": deepcopy(reset_info),
                        "collection_cutoff": False,
                    },
                )

    # Keep a final partial episode instead of silently losing experience.
    if trajectory.num_steps > 0 and (not trajectories or trajectories[-1] is not trajectory):
        trajectory.metadata["collection_cutoff"] = not trajectory.done
        trajectories.append(trajectory)

    return trajectories


def split_trajectories(
    trajectories: list[Trajectory],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Split whole trajectories to avoid train/validation temporal leakage."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    indices = list(range(len(trajectories)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * train_fraction)

    if split == 0 or split == len(indices):
        raise ValueError(
            "Not enough trajectories for a non-empty train/validation split. "
            f"Got {len(trajectories)} trajectories."
        )

    train = [trajectories[i] for i in indices[:split]]
    val = [trajectories[i] for i in indices[split:]]
    return train, val


@dataclass
class TemporalSample:
    """Model-independent sequence after temporal sampling."""

    observations: list[Any]
    action_chunks: list[list[Any]]
    reward_chunks: list[list[float]]
    terminated: list[bool]
    truncated: list[bool]
    infos: list[list[dict[str, Any]]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_lengths(self) -> list[int]:
        return [len(chunk) for chunk in self.action_chunks]


class TemporalSequenceDataset(Dataset):
    """
    Convert raw trajectories to model-time sequences.

    frame_skip is the number of raw environment transitions between two model
    observations. Actions in between are kept as an action chunk.
    """

    def __init__(
        self,
        trajectories: list[Trajectory],
        *,
        history_size: int,
        frame_skip: int,
        window_stride: int | None = None,
        allow_partial_final_chunk: bool = False,
    ):
        if history_size < 1:
            raise ValueError("history_size must be >= 1")
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")

        self.trajectories = trajectories
        self.history_size = history_size
        self.frame_skip = frame_skip
        self.window_stride = window_stride or frame_skip
        self.allow_partial_final_chunk = allow_partial_final_chunk

        if self.window_stride < 1:
            raise ValueError("window_stride must be >= 1")

        self._indices = self._build_index()

    def _chunk_ends(self, trajectory_index: int, start: int) -> list[int] | None:
        trajectory = self.trajectories[trajectory_index]
        cursor = start
        ends: list[int] = []

        for _ in range(self.history_size):
            remaining = trajectory.num_steps - cursor
            if remaining <= 0:
                return None

            chunk_len = min(self.frame_skip, remaining)
            end = cursor + chunk_len

            if chunk_len < self.frame_skip:
                if not self.allow_partial_final_chunk:
                    return None
                if end != trajectory.num_steps or not trajectory.done:
                    return None

            ends.append(end)
            cursor = end

        return ends

    def _build_index(self) -> list[tuple[int, int]]:
        indices: list[tuple[int, int]] = []
        for trajectory_index, trajectory in enumerate(self.trajectories):
            for start in range(0, trajectory.num_steps, self.window_stride):
                if self._chunk_ends(trajectory_index, start) is not None:
                    indices.append((trajectory_index, start))
        return indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> TemporalSample:
        trajectory_index, start = self._indices[index]
        trajectory = self.trajectories[trajectory_index]
        ends = self._chunk_ends(trajectory_index, start)
        if ends is None:
            raise RuntimeError("Temporal dataset index became invalid")

        observations = [trajectory.observations[start]]
        action_chunks: list[list[Any]] = []
        reward_chunks: list[list[float]] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        info_chunks: list[list[dict[str, Any]]] = []

        cursor = start
        for end in ends:
            action_chunks.append(trajectory.actions[cursor:end])
            reward_chunks.append(trajectory.rewards[cursor:end])
            info_chunks.append(trajectory.infos[cursor:end])
            terminated.append(trajectory.terminated[end - 1])
            truncated.append(trajectory.truncated[end - 1])
            observations.append(trajectory.observations[end])
            cursor = end

        return TemporalSample(
            observations=observations,
            action_chunks=action_chunks,
            reward_chunks=reward_chunks,
            terminated=terminated,
            truncated=truncated,
            infos=info_chunks,
            metadata={
                "trajectory_index": trajectory_index,
                "trajectory_metadata": trajectory.metadata,
                "raw_start_step": start,
                "frame_skip": self.frame_skip,
                "chunk_lengths": [len(chunk) for chunk in action_chunks],
            },
        )
