import math
import os
import random
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data import collect_trajectories, split_trajectories, TemporalSequenceDataset
from src.environments.gymnasium import GymnasiumEnvironment
from src.evaluation.world_model_metrics import (evaluate_world_model, flip_binary_centered_actions, )
from src.lewm import LeWMDataset, build_lewm, compute_lewm_loss
from src.interfaces.environment import EnvironmentInterface


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


def make_environment(cfg: DictConfig) -> EnvironmentInterface:
  if cfg.backend != "gymnasium":
    raise NotImplementedError(f"Environment backend '{cfg.backend}' is not implemented yet")
  return GymnasiumEnvironment(env_id=cfg.env_id, observation_mode=cfg.observation_mode, )


def make_lewm_dataset(temporal_dataset: TemporalSequenceDataset, cfg: DictConfig, frame_skip: int, ) -> LeWMDataset:
  action_cfg = cfg.environment.action
  return LeWMDataset(temporal_dataset, image_size=cfg.model.image_size, frame_skip=frame_skip, action_type=action_cfg.type, action_encoding=action_cfg.get("encoding"), num_actions=action_cfg.get("num_actions"),
                     action_dim=action_cfg.get("action_dim"), normalization=cfg.model.preprocessing.normalization, interpolation=cfg.model.preprocessing.interpolation,
                     )


@torch.no_grad()
def validate(model, sigreg, loader, *, history_size: int, sigreg_weight: float, device: torch.device, ) -> dict[str, float]:
  model.eval()
  totals = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}

  for batch in loader:
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda", ):
      loss, pred_loss, sigreg_loss = compute_lewm_loss(model, sigreg, batch, history_size=history_size, sigreg_weight=sigreg_weight, device=device, )
    totals["loss"] += loss.item()
    totals["pred_loss"] += pred_loss.item()
    totals["sigreg_loss"] += sigreg_loss.item()

  n = len(loader)
  return {key: value / n for key, value in totals.items()}


def get_counterfactual_fn(cfg: DictConfig):
  name = cfg.environment.get("counterfactual", None)
  if name is None:
    return None
  if name == "flip_binary_centered":
    return flip_binary_centered_actions
  raise ValueError(f"Unknown counterfactual evaluator: {name}")


@hydra.main(version_base=None, config_path="../config/pipeline", config_name="config", )
def main(cfg: DictConfig) -> None:
  if cfg.model.name != "lewm":
    raise NotImplementedError(f"Model '{cfg.model.name}' is not implemented yet. "
                              "Add a model-specific dataset/runtime while keeping src.data unchanged.")

  set_seed(cfg.seed)
  device = get_device()

  print("=" * 72)
  print(f"{cfg.model.name} on {cfg.environment.name}")
  print("=" * 72)
  print("Device:", device)
  if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

  # 1) Environment -> raw trajectories.
  env = make_environment(cfg.environment)
  num_env_steps = int(cfg.environment.collection.num_env_steps)
  print(f"Collecting {num_env_steps} raw environment steps...")
  trajectories = collect_trajectories(env, num_env_steps=num_env_steps, seed=cfg.seed)
  env.close()

  num_raw_steps = sum(t.num_steps for t in trajectories)
  print(f"Collected {num_raw_steps} raw steps in {len(trajectories)} trajectories.")
  train_trajectories, val_trajectories = split_trajectories(trajectories, train_fraction=cfg.data.train_fraction, seed=cfg.seed, )

  # 2) Raw trajectories -> generic temporal samples. frame_skip lives here.
  frame_skip = int(cfg.environment.temporal.frame_skip)
  temporal_kwargs = dict(history_size=int(cfg.data.history_size), frame_skip=frame_skip, window_stride=cfg.environment.temporal.get("window_stride", None), allow_partial_final_chunk=bool(cfg.data.allow_partial_final_chunk), )
  train_temporal = TemporalSequenceDataset(train_trajectories, **temporal_kwargs)
  val_temporal = TemporalSequenceDataset(val_trajectories, **temporal_kwargs)

  # 3) Generic temporal samples -> vanilla LeWM tensors.
  train_dataset = make_lewm_dataset(train_temporal, cfg, frame_skip)
  val_dataset = make_lewm_dataset(val_temporal, cfg, frame_skip)

  if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise RuntimeError(f"No usable temporal samples: train={len(train_dataset)}, "
                       f"val={len(val_dataset)}. Check collection size/frame_skip/history_size.")

  sample = train_dataset[0]
  print("Training samples:", len(train_dataset))
  print("Validation samples:", len(val_dataset))
  print("Example pixels:", tuple(sample["pixels"].shape))
  print("Example actions:", tuple(sample["action"].shape))
  print("LeWM action input dim:", train_dataset.action_input_dim)

  generator = torch.Generator().manual_seed(cfg.seed)
  train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True, num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda"), generator=generator, )
  val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda"), )

  # 4) Build and train LeWM.
  model, sigreg = build_lewm(cfg.model, action_input_dim=train_dataset.action_input_dim, history_size=cfg.data.history_size, )
  model = model.to(device)
  sigreg = sigreg.to(device)
  print("Trainable parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}", )

  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay, )

  total_steps = cfg.training.epochs * len(train_loader)
  warmup_steps = max(1, int(cfg.training.warmup_fraction * total_steps))

  def lr_schedule(step: int) -> float:
    if step < warmup_steps:
      return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
  sigreg_weight = float(cfg.model.loss.sigreg.weight)
  counterfactual_fn = get_counterfactual_fn(cfg)

  checkpoint_dir = (Path(os.environ.get("JEPA_DATA_ROOT", "data")) / "checkpoints" / f"{cfg.environment.name}_{cfg.model.name}")
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  best_val_loss = float("inf")

  for epoch in range(1, cfg.training.epochs + 1):
    model.train()
    running = {"loss": 0.0, "pred": 0.0, "sigreg": 0.0}

    for batch_idx, batch in enumerate(train_loader):
      optimizer.zero_grad(set_to_none=True)
      with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda", ):
        loss, pred_loss, sigreg_loss = compute_lewm_loss(model, sigreg, batch, history_size=cfg.data.history_size, sigreg_weight=sigreg_weight, device=device, )

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.gradient_clip)
      optimizer.step()
      scheduler.step()

      running["loss"] += loss.item()
      running["pred"] += pred_loss.item()
      running["sigreg"] += sigreg_loss.item()

      if (cfg.training.progress_every_n_batches > 0 and batch_idx % cfg.training.progress_every_n_batches == 0):
        print(f"  epoch {epoch:03d} | batch {batch_idx:04d}/{len(train_loader)}")

    n_train = len(train_loader)
    val_metrics = validate(model, sigreg, val_loader, history_size=cfg.data.history_size, sigreg_weight=sigreg_weight, device=device, )

    wm_metrics = None
    if epoch % cfg.training.metrics_every_n_epochs == 0:
      wm_metrics = evaluate_world_model(model=model, dataloader=val_loader, device=device, counterfactual_action_fn=counterfactual_fn, )

    print(f"Epoch {epoch:03d}/{cfg.training.epochs} | "
          f"lr={optimizer.param_groups[0]['lr']:.2e} | "
          f"train loss={running['loss'] / n_train:.4f} | "
          f"train pred={running['pred'] / n_train:.4f} | "
          f"train sigreg={running['sigreg'] / n_train:.4f} | "
          f"val loss={val_metrics['loss']:.4f} | "
          f"val pred={val_metrics['pred_loss']:.4f} | "
          f"val sigreg={val_metrics['sigreg_loss']:.4f}")

    if wm_metrics is not None:
      print("    WM metrics | "
            f"final_pred={wm_metrics['final_prediction_mse']:.4f} | "
            f"copy={wm_metrics['copy_mse']:.4f} | "
            f"copy gain={100 * wm_metrics['improvement_over_copy']:.1f}% | "
            f"shuffled={wm_metrics['shuffled_action_mse']:.4f} | "
            f"action gain={100 * wm_metrics['action_gain']:.1f}% | "
            f"beats shuffled={100 * wm_metrics['correct_beats_shuffled']:.1f}% | "
            f"retrieval={100 * wm_metrics['retrieval_top1']:.1f}% | "
            f"latent std={wm_metrics['latent_feature_std']:.3f}")
      if "counterfactual_mse" in wm_metrics:
        print("               "
              f"counterfactual={wm_metrics['counterfactual_mse']:.4f} | "
              f"gain={100 * wm_metrics['counterfactual_gain']:.1f}% | "
              f"beats={100 * wm_metrics['correct_beats_counterfactual']:.1f}%")

    if val_metrics["loss"] < best_val_loss:
      best_val_loss = val_metrics["loss"]
      path = checkpoint_dir / "best.pt"
      torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_loss": best_val_loss, "config": OmegaConf.to_container(cfg, resolve=True), }, path, )
      print(f"  -> saved best model to {path}")

  print("Training finished")
  print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
  main()
