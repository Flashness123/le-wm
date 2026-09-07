from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional
from torch.utils.data import Dataset
from src.interfaces.environment_interface import EnvironmentInterface
from src.environments.gymnasium_env import GymnasiumEnvironment

import numpy as np
import torch
import torch.nn.functional as F
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
    frame_skip: int = 1,
) -> list[Transition]:
    """
    Collect world-model transitions.

    One stored transition contains exactly `frame_skip`
    environment actions:

        observation
            -> action_0
            -> action_1
            -> ...
            -> action_(frame_skip - 1)
            -> next_observation

    The complete action chunk is stored as one flattened vector.

    Example for CartPole with frame_skip=4:

        action = [0, 1, 1, 0]

    If an episode terminates before all `frame_skip` actions have
    been executed, that incomplete transition is discarded.
    """

    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1")

    transitions: list[Transition] = []

    observation, _ = env.reset(seed=seed)

    episode_id = 0
    timestep = 0

    while len(transitions) < num_steps:

        start_observation = deepcopy(observation)

        actions = []
        total_reward = 0.0

        result = None

        # --------------------------------------------------
        # Execute exactly frame_skip environment steps
        # --------------------------------------------------

        for _ in range(frame_skip):

            action = env.sample_action()

            result = env.step(action)

            actions.append(
                np.asarray(
                    deepcopy(action),
                    dtype=np.float32,
                )
            )

            total_reward += result.reward

            if result.done:
                break

        # --------------------------------------------------
        # Store only COMPLETE action chunks
        # --------------------------------------------------

        if len(actions) == frame_skip:

            # Examples:
            #
            # CartPole, frame_skip=4:
            #     (4,)
            #
            # Continuous action_dim=2, frame_skip=4:
            #     (8,)
            #
            action_chunk = np.stack(
                actions,
                axis=0,
            ).reshape(-1)

            transitions.append(
                Transition(
                    observation=start_observation,
                    action=action_chunk,
                    next_observation=deepcopy(
                        result.observation
                    ),
                    reward=total_reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    episode_id=episode_id,
                    timestep=timestep,
                )
            )

            timestep += 1

        # --------------------------------------------------
        # Episode handling
        # --------------------------------------------------

        if result.done:

            episode_id += 1
            timestep = 0

            observation, _ = env.reset()

        else:

            observation = result.observation

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
    Converts temporal transition sequences into tensors for LeWM.

    Supports:

        continuous
            Raw continuous action vectors.

        binary_centered
            CartPole-style binary actions:
                0 -> -1
                1 -> +1

        one_hot
            Generic discrete actions:
                action 0 -> [1, 0, 0, ...]
                action 1 -> [0, 1, 0, ...]
                ...

    For frame_skip > 1, each individual action in the action
    chunk is encoded and the resulting vectors are flattened.

    Example:
        frame_skip = 2
        num_actions = 3
        action chunk = [0, 2]

        becomes:

        [1, 0, 0, 0, 0, 1]
    """

    def __init__(
        self,
        sequence_dataset: TransitionSequenceDataset,
        image_size: int = 224,
        action_encoding: str = "continuous",
        num_actions: int | None = None,
    ):
        self.sequence_dataset = sequence_dataset
        self.image_size = image_size
        self.action_encoding = action_encoding
        self.num_actions = num_actions

        if (
            self.action_encoding == "one_hot"
            and self.num_actions is None
        ):
            raise ValueError(
                "num_actions must be provided "
                "when action_encoding='one_hot'"
            )

    def __len__(self):
        return len(self.sequence_dataset)

    def _get_pixels(
        self,
        observation,
    ) -> np.ndarray:
        """
        Supports both:

            CartPole:
                {
                    "pixels": ...,
                    "state": ...
                }

            MiniGrid:
                direct RGB ndarray
        """

        if isinstance(observation, np.ndarray):
            return observation

        if isinstance(observation, dict):
            if "pixels" in observation:
                return observation["pixels"]

        raise ValueError(
            "Could not find pixel observation. "
            f"Got type: {type(observation)}"
        )

    def _encode_actions(
        self,
        actions,
    ) -> torch.Tensor:

        # Usually:
        #
        # (history_size, frame_skip)
        #
        raw_actions = np.stack(
            actions,
            axis=0,
        )

        if raw_actions.ndim == 1:
            raw_actions = raw_actions[:, None]

        # --------------------------------------------------
        # Continuous actions
        # --------------------------------------------------

        if self.action_encoding == "continuous":

            return torch.from_numpy(
                raw_actions.astype(np.float32)
            )

        # --------------------------------------------------
        # CartPole binary actions
        #
        # 0 -> -1
        # 1 -> +1
        # --------------------------------------------------

        if self.action_encoding == "binary_centered":

            actions_tensor = torch.from_numpy(
                raw_actions.astype(np.float32)
            )

            return actions_tensor * 2.0 - 1.0

        # --------------------------------------------------
        # Generic discrete one-hot actions
        # --------------------------------------------------

        if self.action_encoding == "one_hot":

            action_ids = torch.from_numpy(
                raw_actions.astype(np.int64)
            )

            one_hot = F.one_hot(
                action_ids,
                num_classes=self.num_actions,
            ).float()

            # Example:
            #
            # history_size = 3
            # frame_skip   = 1
            # num_actions  = 7
            #
            # (3, 1, 7)
            #      ↓
            # (3, 7)
            #
            # With frame_skip = 4:
            #
            # (3, 4, 7)
            #      ↓
            # (3, 28)

            return one_hot.reshape(
                one_hot.shape[0],
                -1,
            )

        raise ValueError(
            f"Unknown action encoding: "
            f"{self.action_encoding}"
        )

    def __getitem__(
        self,
        index,
    ):

        sample = self.sequence_dataset[index]

        history = sample["observations"]
        target = sample["target_observation"]

        observations_with_target = [
            *history,
            target,
        ]

        # --------------------------------------------------
        # Pixels
        # --------------------------------------------------

        pixels = torch.stack([
            preprocess_lewm_frame(
                self._get_pixels(observation),
                image_size=self.image_size,
            )
            for observation
            in observations_with_target
        ])

        # --------------------------------------------------
        # Actions
        # --------------------------------------------------

        actions = self._encode_actions(
            sample["actions"]
        )

        result = {
            "pixels": pixels,
            "action": actions,
        }

        # --------------------------------------------------
        # Optional numeric ground-truth state
        #
        # CartPole has this.
        # MiniGrid pixel mode doesn't need it.
        # --------------------------------------------------

        numeric_states = []

        for observation in observations_with_target:

            if (
                isinstance(observation, dict)
                and "state" in observation
                and isinstance(
                    observation["state"],
                    np.ndarray,
                )
            ):
                numeric_states.append(
                    torch.as_tensor(
                        observation["state"],
                        dtype=torch.float32,
                    )
                )

            else:
                numeric_states = []
                break

        if numeric_states:
            result["state"] = torch.stack(
                numeric_states
            )

        return result