from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from stable_pretraining.backbone.utils import vit_hf
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from src.data import TemporalSequenceDataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class LeWMDataset(Dataset):
    """Convert generic temporal samples into the tensors vanilla RGB LeWM expects."""

    def __init__(
        self,
        temporal_dataset: TemporalSequenceDataset,
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
        self.temporal_dataset = temporal_dataset
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
                    "Discrete LeWM actions require 'one_hot' or 'centered_binary' encoding"
                )
            if action_encoding == "centered_binary" and num_actions != 2:
                raise ValueError("centered_binary is only valid for two actions")
        elif action_type == "continuous":
            if action_dim is None:
                raise ValueError("action_dim is required for continuous actions")
        else:
            raise ValueError("action_type must be 'discrete' or 'continuous'")

    def __len__(self) -> int:
        return len(self.temporal_dataset)

    @property
    def action_input_dim(self) -> int:
        if self.action_type == "discrete":
            if self.action_encoding == "one_hot":
                return self.frame_skip * int(self.num_actions)
            return self.frame_skip
        return self.frame_skip * int(self.action_dim)

    @staticmethod
    def _extract_pixels(observation: Any) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation
        if isinstance(observation, dict) and "pixels" in observation:
            return observation["pixels"]
        raise TypeError(
            "LeWMDataset expected an RGB ndarray or a dict containing 'pixels', "
            f"got {type(observation)}"
        )

    def _process_frame(self, observation: Any) -> torch.Tensor:
        frame = self._extract_pixels(observation)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB frame, got shape {frame.shape}")

        x = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
        x = TF.resize(
            x,
            [self.image_size, self.image_size],
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
                f"Vanilla LeWM requires {self.frame_skip} actions per chunk; "
                f"got {len(chunk)}"
            )

        if self.action_type == "discrete":
            ids = torch.as_tensor(np.asarray(chunk, dtype=np.int64).reshape(-1), dtype=torch.long)
            if self.action_encoding == "one_hot":
                return F.one_hot(ids, num_classes=int(self.num_actions)).float().reshape(-1)
            return ids.float() * 2.0 - 1.0

        vectors = [
            torch.as_tensor(np.asarray(action), dtype=torch.float32).reshape(-1)
            for action in chunk
        ]
        encoded = torch.cat(vectors)
        if encoded.numel() != self.action_input_dim:
            raise ValueError(
                f"Continuous action chunk has {encoded.numel()} values; "
                f"expected {self.action_input_dim}"
            )
        return encoded

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.temporal_dataset[index]
        result: dict[str, Any] = {
            "pixels": torch.stack([self._process_frame(obs) for obs in sample.observations]),
            "action": torch.stack(
                [self._encode_action_chunk(chunk) for chunk in sample.action_chunks]
            ),
            "reward": torch.tensor(
                [sum(chunk) for chunk in sample.reward_chunks], dtype=torch.float32
            ),
            "terminated": torch.tensor(sample.terminated, dtype=torch.bool),
            "truncated": torch.tensor(sample.truncated, dtype=torch.bool),
        }

        # Optional native state, used only for diagnostics (e.g. CartPole).
        states = []
        for observation in sample.observations:
            if isinstance(observation, dict) and isinstance(observation.get("state"), np.ndarray):
                states.append(torch.as_tensor(observation["state"], dtype=torch.float32))
            else:
                states = []
                break
        if states:
            result["state"] = torch.stack(states)

        return result


def build_lewm(
    cfg: DictConfig,
    *,
    action_input_dim: int,
    history_size: int,
) -> tuple[JEPA, SIGReg]:
    encoder = vit_hf(
        size=cfg.encoder.size,
        patch_size=cfg.encoder.patch_size,
        image_size=cfg.image_size,
        pretrained=cfg.encoder.pretrained,
        use_mask_token=False,
    )

    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.embed_dim,
        output_dim=cfg.embed_dim,
        depth=cfg.predictor.depth,
        heads=cfg.predictor.heads,
        mlp_dim=cfg.predictor.mlp_dim,
        dim_head=cfg.predictor.dim_head,
        dropout=cfg.predictor.dropout,
        emb_dropout=cfg.predictor.emb_dropout,
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=Embedder(input_dim=action_input_dim, emb_dim=cfg.embed_dim),
        projector=MLP(
            input_dim=cfg.embed_dim,
            hidden_dim=cfg.projector.hidden_dim,
            output_dim=cfg.embed_dim,
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=cfg.embed_dim,
            hidden_dim=cfg.projector.hidden_dim,
            output_dim=cfg.embed_dim,
            norm_fn=nn.BatchNorm1d,
        ),
    )

    return model, SIGReg(**dict(cfg.loss.sigreg.kwargs))


def compute_lewm_loss(
    model: JEPA,
    sigreg: SIGReg,
    batch: dict[str, torch.Tensor],
    *,
    history_size: int,
    sigreg_weight: float,
    device: torch.device,
):
    pixels = batch["pixels"].to(device, non_blocking=True)
    actions = batch["action"].to(device, non_blocking=True)

    output = model.encode({"pixels": pixels, "action": actions})
    emb = output["emb"]
    act_emb = output["act_emb"]

    predicted_emb = model.predict(
        emb[:, :history_size],
        act_emb[:, :history_size],
    )
    target_emb = emb[:, 1:]

    pred_loss = (predicted_emb - target_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss = pred_loss + sigreg_weight * sigreg_loss
    return loss, pred_loss, sigreg_loss
