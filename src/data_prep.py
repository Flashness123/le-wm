from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional
from torch.utils.data import Dataset
from src.environment_interface import EnvironmentInterface
from src.environments.gymnasium_env import GymnasiumEnvironment

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)


def preprocess_lewm_frame(
    frame: np.ndarray,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Convert a Gymnasium RGB frame:

        (H, W, 3), uint8, [0, 255]

    into the image representation expected by LeWM:

        (3, 224, 224), float32, ImageNet-normalized
    """

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), got {frame.shape}"
        )

    # HWC uint8 -> CHW float32
    x = torch.from_numpy(
        np.ascontiguousarray(frame)
    ).permute(2, 0, 1).float()

    # [0, 255] -> [0, 1]
    x = x / 255.0

    # CartPole is 400x600. For now deliberately make the
    # same square input size expected by LeWM's ViT.
    x = TF.resize(
        x,
        size=[image_size, image_size],
        antialias=True,
    )

    # ImageNet normalization used by LeWM
    x = (x - IMAGENET_MEAN) / IMAGENET_STD

    return x

@dataclass
class Transition:
    observation: Any
    action: Any
    next_observation: Any

    reward: float

    terminated: bool
    truncated: bool

    episode_id: int
    timestep: int

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


def collect_transitions(
    env: EnvironmentInterface,
    num_steps: int,
    seed: Optional[int] = None,
) -> list[Transition]:

    transitions = []

    observation, _ = env.reset(seed=seed)

    episode_id = 0
    timestep = 0

    for _ in range(num_steps):

        # For now: random exploration
        action = env.sample_action()

        result = env.step(action)

        transition = Transition(
            observation=deepcopy(observation),
            action=deepcopy(action),
            next_observation=deepcopy(result.observation),

            reward=result.reward,

            terminated=result.terminated,
            truncated=result.truncated,

            episode_id=episode_id,
            timestep=timestep,
        )

        transitions.append(transition)

        observation = result.observation
        timestep += 1

        if result.done:

            episode_id += 1
            timestep = 0

            observation, _ = env.reset()

    return transitions

class TransitionSequenceDataset(Dataset):
    """
    Converts sequential transitions into temporal windows.

    For history_size=3:

        observations = [s_t, s_t+1, s_t+2]
        actions      = [a_t, a_t+1, a_t+2]
        target       = s_t+3

    Sequences are never allowed to cross episode boundaries.
    """

    def __init__(
        self,
        transitions: list[Transition],
        history_size: int = 3,
    ):
        self.transitions = transitions
        self.history_size = history_size

        self.valid_starts = self._find_valid_starts()

    def _find_valid_starts(self) -> list[int]:

        valid_starts = []

        for start in range(
            len(self.transitions) - self.history_size
        ):
            sequence = self.transitions[
                start : start + self.history_size
            ]

            target_transition = self.transitions[
                start + self.history_size
            ]

            episode_id = sequence[0].episode_id

            same_episode = all(
                transition.episode_id == episode_id
                for transition in sequence
            )

            same_episode = (
                same_episode
                and target_transition.episode_id == episode_id
            )

            if same_episode:
                valid_starts.append(start)

        return valid_starts

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, index: int):

        start = self.valid_starts[index]

        sequence = self.transitions[
            start : start + self.history_size
        ]

        observations = [
            transition.observation
            for transition in sequence
        ]

        actions = [
            transition.action
            for transition in sequence
        ]

        target_observation = sequence[-1].next_observation

        return {
            "observations": observations,
            "actions": actions,
            "target_observation": target_observation,
        }

class LeWMPixelSequenceDataset(Dataset):
    """
    Converts TransitionSequenceDataset samples into tensors
    directly usable by the visual LeWM pipeline.

    For history_size=3:

        pixels:
            [s0, s1, s2, s3]
            shape = (4, 3, 224, 224)

        actions:
            [a0, a1, a2]
            shape = (3, 1)

        states:
            hidden CartPole states, evaluation only
            shape = (4, 4)

    The model must NOT use `states` during JEPA training.
    """

    def __init__(
        self,
        sequence_dataset: TransitionSequenceDataset,
        image_size: int = 224,
    ):
        self.sequence_dataset = sequence_dataset
        self.image_size = image_size

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, index):

        sample = self.sequence_dataset[index]

        history = sample["observations"]
        target = sample["target_observation"]

        # --------------------------------------------------
        # Pixels
        # --------------------------------------------------

        observations_with_target = [
            *history,
            target,
        ]

        pixels = torch.stack([
            preprocess_lewm_frame(
                observation["pixels"],
                image_size=self.image_size,
            )
            for observation in observations_with_target
        ])

        # shape:
        # (history_size + 1, 3, 224, 224)

        # --------------------------------------------------
        # Actions
        # --------------------------------------------------

        actions = torch.tensor(
            sample["actions"],
            dtype=torch.float32,
        ).unsqueeze(-1)

        # shape:
        # (history_size, 1)

        # Map CartPole:
        #
        # 0 -> -1
        # 1 -> +1
        #
        # This produces a centered scalar action,
        # similar in spirit to the normalized action input
        # used in the original LeWM training pipeline.
        actions = actions * 2.0 - 1.0

        # --------------------------------------------------
        # Ground-truth state
        # evaluation only
        # --------------------------------------------------

        states = torch.stack([
            torch.as_tensor(
                observation["state"],
                dtype=torch.float32,
            )
            for observation in observations_with_target
        ])

        return {
            "pixels": pixels,
            "action": actions,
            "state": states,
        }