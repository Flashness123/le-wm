from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from src.data.temporal_dataset import TemporalSample
from src.interfaces.model_adapter_interface import ModelAdapterInterface


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(3, 1, 1)
IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(3, 1, 1)


class LeWMRGBAdapter(ModelAdapterInterface):
    """
    Current LeWM RGB baseline adapter.

    This is deliberately MODEL-SPECIFIC. It owns:
      - pixel extraction
      - 224x224 image conversion
      - ImageNet normalization
      - LeWM-compatible fixed-width action vectors

    A future categorical ARC-JEPA should use a different adapter rather than
    changing the generic trajectory / temporal layers.
    """

    def __init__(
        self,
        *,
        image_size: int,
        frame_skip: int,
        action_type: str,
        action_encoding: str | None = None,
        num_actions: int | None = None,
        action_dim: int | None = None,
        normalization: str = "imagenet",
        interpolation: str = "bilinear",
    ):
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.action_type = action_type
        self.action_encoding = action_encoding
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.normalization = normalization

        interpolation_map = {
            "bilinear": InterpolationMode.BILINEAR,
            "nearest": InterpolationMode.NEAREST,
        }
        if interpolation not in interpolation_map:
            raise ValueError("interpolation must be 'bilinear' or 'nearest'")
        self.interpolation = interpolation_map[interpolation]

        if action_type == "discrete":
            if num_actions is None:
                raise ValueError("num_actions is required for discrete actions")
            if action_encoding not in {"one_hot", "centered_binary"}:
                raise ValueError(
                    "Discrete LeWM actions require action_encoding='one_hot' "
                    "or 'centered_binary'"
                )
            if action_encoding == "centered_binary" and num_actions != 2:
                raise ValueError("centered_binary is only valid for two actions")

        elif action_type == "continuous":
            if action_dim is None:
                raise ValueError("action_dim is required for continuous actions")

        else:
            raise ValueError("action_type must be 'discrete' or 'continuous'")

    @property
    def action_input_dim(self) -> int:
        if self.action_type == "discrete":
            if self.action_encoding == "one_hot":
                return self.frame_skip * int(self.num_actions)
            return self.frame_skip

        return self.frame_skip * int(self.action_dim)

    def _extract_pixels(self, observation: Any) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation

        if isinstance(observation, dict) and "pixels" in observation:
            return observation["pixels"]

        raise TypeError(
            "LeWMRGBAdapter expected an RGB ndarray or a dict containing "
            f"'pixels', got {type(observation)}"
        )

    def _process_frame(self, observation: Any) -> torch.Tensor:
        frame = self._extract_pixels(observation)

        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB frame, got shape {frame.shape}")

        x = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float()
        x = x / 255.0

        # Kept intentionally compatible with the current LeWM baseline.
        # ARC-specific encoders should not be forced through this transform.
        x = TF.resize(
            x,
            size=[self.image_size, self.image_size],
            interpolation=self.interpolation,
            antialias=(self.interpolation == InterpolationMode.BILINEAR),
        )

        if self.normalization == "imagenet":
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        elif self.normalization != "none":
            raise ValueError("normalization must be 'imagenet' or 'none'")

        return x

    def _encode_action_chunk(self, chunk: list[Any]) -> torch.Tensor:
        if len(chunk) != self.frame_skip:
            raise ValueError(
                "Vanilla LeWM requires fixed-size action chunks. "
                f"Expected {self.frame_skip}, got {len(chunk)}. "
                "Keep allow_partial_final_chunk=false for this adapter."
            )

        if self.action_type == "discrete":
            ids = torch.as_tensor(
                np.asarray(chunk, dtype=np.int64).reshape(-1),
                dtype=torch.long,
            )

            if self.action_encoding == "one_hot":
                return F.one_hot(
                    ids,
                    num_classes=int(self.num_actions),
                ).float().reshape(-1)

            return ids.float() * 2.0 - 1.0

        vectors = [
            torch.as_tensor(np.asarray(action), dtype=torch.float32).reshape(-1)
            for action in chunk
        ]
        encoded = torch.cat(vectors, dim=0)

        expected = self.action_input_dim
        if encoded.numel() != expected:
            raise ValueError(
                f"Continuous action chunk has {encoded.numel()} values; "
                f"expected {expected}."
            )

        return encoded

    def adapt(self, sample: TemporalSample) -> dict[str, Any]:
        pixels = torch.stack(
            [self._process_frame(obs) for obs in sample.observations],
            dim=0,
        )
        actions = torch.stack(
            [self._encode_action_chunk(chunk) for chunk in sample.action_chunks],
            dim=0,
        )

        rewards = torch.tensor(
            [sum(chunk) for chunk in sample.reward_chunks],
            dtype=torch.float32,
        )
        terminated = torch.tensor(sample.terminated, dtype=torch.bool)
        truncated = torch.tensor(sample.truncated, dtype=torch.bool)

        result: dict[str, Any] = {
            "pixels": pixels,
            "action": actions,
            "reward": rewards,
            "terminated": terminated,
            "truncated": truncated,
        }

        # Optional CartPole-style hidden state for diagnostics only.
        numeric_states = []
        for observation in sample.observations:
            if (
                isinstance(observation, dict)
                and "state" in observation
                and isinstance(observation["state"], np.ndarray)
            ):
                numeric_states.append(
                    torch.as_tensor(observation["state"], dtype=torch.float32)
                )
            else:
                numeric_states = []
                break

        if numeric_states:
            result["state"] = torch.stack(numeric_states)

        return result
