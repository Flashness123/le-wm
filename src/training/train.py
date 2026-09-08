import math
import os
import random
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.collector import collect_trajectories
from src.data.model_dataset import ModelDataset
from src.data.source import split_by_trajectory
from src.data.temporal_dataset import TemporalSequenceDataset
from src.environments.factory import build_environment
from src.evaluation.world_model_metrics import (
    evaluate_world_model,
    flip_binary_centered_actions,
)
from src.models.adapters.lewm import LeWMRGBAdapter
from src.models.lewm_runtime import build_lewm, compute_lewm_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_adapter(cfg: DictConfig, frame_skip: int):
    if cfg.model.name != "lewm":
        raise NotImplementedError(
            f"Model '{cfg.model.name}' is not implemented yet. "
            "Add a model-specific adapter/runtime without changing the raw data pipeline."
        )

    if cfg.model.observation_adapter != "rgb":
        raise NotImplementedError(
            "This refactor includes only the current LeWM RGB baseline adapter. "
            "A future categorical ARC model should get its own adapter."
        )

    action_cfg = cfg.environment.action

    return LeWMRGBAdapter(
        image_size=cfg.model.image_size,
        frame_skip=frame_skip,
        action_type=action_cfg.type,
        action_encoding=action_cfg.get("encoding"),
        num_actions=action_cfg.get("num_actions"),
        action_dim=action_cfg.get("action_dim"),
        normalization=cfg.model.preprocessing.normalization,
        interpolation=cfg.model.preprocessing.interpolation,
    )


@torch.no_grad()
def validate(
    model,
    sigreg,
    loader,
    *,
    history_size: int,
    sigreg_weight: float,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    total_pred = 0.0
    total_sigreg = 0.0
    num_batches = 0

    for batch in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, pred_loss, sigreg_loss = compute_lewm_loss(
                model,
                sigreg,
                batch,
                history_size=history_size,
                sigreg_weight=sigreg_weight,
                device=device,
            )

        total_loss += loss.item()
        total_pred += pred_loss.item()
        total_sigreg += sigreg_loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "pred_loss": total_pred / num_batches,
        "sigreg_loss": total_sigreg / num_batches,
    }


def get_counterfactual_fn(cfg: DictConfig):
    name = cfg.environment.get("counterfactual", None)

    if name is None:
        return None
    if name == "flip_binary_centered":
        return flip_binary_centered_actions

    raise ValueError(f"Unknown counterfactual evaluator: {name}")


@hydra.main(
    version_base=None,
    config_path="../../config/pipeline",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device()

    print("=" * 72)
    print(f"{cfg.model.name} on {cfg.environment.name}")
    print("=" * 72)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # 1. SOURCE-SPECIFIC COLLECTION -> CANONICAL RAW TRAJECTORIES
    # ------------------------------------------------------------------

    env = build_environment(cfg.environment)

    print(
        f"Collecting {cfg.environment.collection.num_env_steps} raw environment steps..."
    )

    source = collect_trajectories(
        env,
        num_env_steps=cfg.environment.collection.num_env_steps,
        seed=cfg.seed,
    )
    env.close()

    print(
        f"Collected {source.num_steps} raw steps in {len(source)} trajectories."
    )

    train_source, val_source = split_by_trajectory(
        source,
        train_fraction=cfg.data.train_fraction,
        seed=cfg.seed,
    )

    # ------------------------------------------------------------------
    # 2. GENERIC TEMPORAL VIEW
    # ------------------------------------------------------------------

    frame_skip = cfg.environment.temporal.frame_skip
    window_stride = cfg.environment.temporal.get("window_stride", None)

    train_temporal = TemporalSequenceDataset(
        train_source,
        history_size=cfg.data.history_size,
        frame_skip=frame_skip,
        window_stride=window_stride,
        allow_partial_final_chunk=cfg.data.allow_partial_final_chunk,
    )
    val_temporal = TemporalSequenceDataset(
        val_source,
        history_size=cfg.data.history_size,
        frame_skip=frame_skip,
        window_stride=window_stride,
        allow_partial_final_chunk=cfg.data.allow_partial_final_chunk,
    )

    # ------------------------------------------------------------------
    # 3. MODEL-SPECIFIC ADAPTER
    # ------------------------------------------------------------------

    adapter = build_model_adapter(cfg, frame_skip=frame_skip)
    train_dataset = ModelDataset(train_temporal, adapter)
    val_dataset = ModelDataset(val_temporal, adapter)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"No usable temporal samples: train={len(train_dataset)}, "
            f"val={len(val_dataset)}. Check collection size/frame_skip/history_size."
        )

    sample = train_dataset[0]
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Example pixels:", tuple(sample["pixels"].shape))
    print("Example actions:", tuple(sample["action"].shape))
    print("LeWM action input dim:", adapter.action_input_dim)

    train_gen = torch.Generator().manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # 4. MODEL-SPECIFIC RUNTIME
    # ------------------------------------------------------------------

    model, sigreg = build_lewm(
        cfg.model,
        action_input_dim=adapter.action_input_dim,
        history_size=cfg.data.history_size,
    )
    model = model.to(device)
    sigreg = sigreg.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = cfg.training.epochs * len(train_loader)
    warmup_steps = max(1, int(cfg.training.warmup_fraction * total_steps))

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_schedule,
    )

    data_root = Path(os.environ.get("JEPA_DATA_ROOT", "data"))
    checkpoint_dir = (
        data_root
        / "checkpoints"
        / f"{cfg.environment.name}_{cfg.model.name}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    sigreg_weight = float(cfg.model.loss.sigreg.weight)
    counterfactual_fn = get_counterfactual_fn(cfg)

    # ------------------------------------------------------------------
    # 5. TRAIN
    # ------------------------------------------------------------------

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()

        running_loss = 0.0
        running_pred = 0.0
        running_sigreg = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss, pred_loss, sigreg_loss = compute_lewm_loss(
                    model,
                    sigreg,
                    batch,
                    history_size=cfg.data.history_size,
                    sigreg_weight=sigreg_weight,
                    device=device,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg.training.gradient_clip,
            )
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_pred += pred_loss.item()
            running_sigreg += sigreg_loss.item()

            if (
                cfg.training.progress_every_n_batches > 0
                and batch_idx % cfg.training.progress_every_n_batches == 0
            ):
                print(
                    f"  epoch {epoch:03d} | "
                    f"batch {batch_idx:04d}/{len(train_loader)}"
                )

        n_train_batches = len(train_loader)
        train_loss = running_loss / n_train_batches
        train_pred = running_pred / n_train_batches
        train_sigreg = running_sigreg / n_train_batches

        val_metrics = validate(
            model,
            sigreg,
            val_loader,
            history_size=cfg.data.history_size,
            sigreg_weight=sigreg_weight,
            device=device,
        )

        wm_metrics = None
        if epoch % cfg.training.metrics_every_n_epochs == 0:
            wm_metrics = evaluate_world_model(
                model=model,
                dataloader=val_loader,
                device=device,
                counterfactual_action_fn=counterfactual_fn,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{cfg.training.epochs} | "
            f"lr={current_lr:.2e} | "
            f"train loss={train_loss:.4f} | "
            f"train pred={train_pred:.4f} | "
            f"train sigreg={train_sigreg:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val pred={val_metrics['pred_loss']:.4f} | "
            f"val sigreg={val_metrics['sigreg_loss']:.4f}"
        )

        if wm_metrics is not None:
            print(
                "    WM metrics | "
                f"final_pred={wm_metrics['final_prediction_mse']:.4f} | "
                f"copy={wm_metrics['copy_mse']:.4f} | "
                f"copy gain={100 * wm_metrics['improvement_over_copy']:.1f}% | "
                f"shuffled={wm_metrics['shuffled_action_mse']:.4f} | "
                f"action gain={100 * wm_metrics['action_gain']:.1f}% | "
                f"beats shuffled={100 * wm_metrics['correct_beats_shuffled']:.1f}% | "
                f"retrieval={100 * wm_metrics['retrieval_top1']:.1f}% | "
                f"latent std={wm_metrics['latent_feature_std']:.3f}"
            )

            if "counterfactual_mse" in wm_metrics:
                print(
                    "               "
                    f"counterfactual={wm_metrics['counterfactual_mse']:.4f} | "
                    f"gain={100 * wm_metrics['counterfactual_gain']:.1f}% | "
                    f"beats={100 * wm_metrics['correct_beats_counterfactual']:.1f}%"
                )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            path = checkpoint_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                path,
            )
            print(f"  -> saved best model to {path}")

    print("Training finished")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()
