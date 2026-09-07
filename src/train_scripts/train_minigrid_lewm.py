import math
import random
from pathlib import Path
import minigrid
import torch
from torch import nn
from torch.utils.data import DataLoader

from stable_pretraining.backbone.utils import vit_hf

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg

from src.environments.gymnasium_env import GymnasiumEnvironment
from src.interfaces.data_interface import (
    collect_transitions,
    TransitionSequenceDataset,
    LeWMPixelSequenceDataset,
)
from src.evaluation.world_model_metrics import (
    evaluate_world_model
)


# ============================================================
# Environment
# ============================================================

ENV_ID = "MiniGrid-DoorKey-8x8-v0"

SEED = 42

NUM_TRANSITIONS = 5_000

# MiniGrid is already turn-based.
# One world-model transition = one environment action.
FRAME_SKIP = 1

NUM_ACTIONS = 7

ACTION_INPUT_DIM = (
    FRAME_SKIP
    * NUM_ACTIONS
)


# ============================================================
# LeWM
# ============================================================

HISTORY_SIZE = 3

IMAGE_SIZE = 224
EMBED_DIM = 192

BATCH_SIZE = 8
EPOCHS = 50

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3

SIGREG_WEIGHT = 0.09

TRAIN_FRACTION = 0.8


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_transitions_by_episode(
    transitions,
    train_fraction: float = 0.8,
    seed: int = 42,
):
    """
    Split entire episodes rather than individual temporal windows.

    This prevents nearly identical adjacent frames from appearing
    in both train and validation sets.
    """

    episode_ids = sorted(
        set(t.episode_id for t in transitions)
    )

    rng = random.Random(seed)
    rng.shuffle(episode_ids)

    split_index = int(
        len(episode_ids) * train_fraction
    )

    train_episode_ids = set(
        episode_ids[:split_index]
    )

    val_episode_ids = set(
        episode_ids[split_index:]
    )

    train_transitions = [
        t for t in transitions
        if t.episode_id in train_episode_ids
    ]

    val_transitions = [
        t for t in transitions
        if t.episode_id in val_episode_ids
    ]

    return train_transitions, val_transitions


# ============================================================
# Build original LeWM architecture
# ============================================================

def build_lewm() -> JEPA:

    # --------------------------------------------------------
    # Original LeWM visual encoder
    # --------------------------------------------------------

    encoder = vit_hf(
        size="tiny",
        patch_size=14,
        image_size=IMAGE_SIZE,
        pretrained=False,
        use_mask_token=False,
    )

    # --------------------------------------------------------
    # Original LeWM action-conditioned predictor
    # --------------------------------------------------------

    predictor = ARPredictor(
        num_frames=HISTORY_SIZE,

        input_dim=EMBED_DIM,
        hidden_dim=EMBED_DIM,
        output_dim=EMBED_DIM,

        depth=6,
        heads=16,

        mlp_dim=2048,
        dim_head=64,

        dropout=0.1,
        emb_dropout=0.0,
    )

    
    action_encoder = Embedder(
        input_dim=ACTION_INPUT_DIM,
        emb_dim=EMBED_DIM,
    )

    # --------------------------------------------------------
    # Original LeWM projector
    # --------------------------------------------------------

    projector = MLP(
        input_dim=EMBED_DIM,
        hidden_dim=2048,
        output_dim=EMBED_DIM,
        norm_fn=nn.BatchNorm1d,
    )

    # --------------------------------------------------------
    # Original prediction projector
    # --------------------------------------------------------

    pred_proj = MLP(
        input_dim=EMBED_DIM,
        hidden_dim=2048,
        output_dim=EMBED_DIM,
        norm_fn=nn.BatchNorm1d,
    )

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


# ============================================================
# LeWM loss
# ============================================================

def compute_loss(
    model,
    sigreg,
    batch,
    device,
):

    pixels = batch["pixels"].to(
        device,
        non_blocking=True,
    )

    actions = batch["action"].to(
        device,
        non_blocking=True,
    )

    # IMPORTANT:
    #
    # pixels:
    #     B x 4 x 3 x 224 x 224
    #
    # actions:
    #     B x 3 x 1

    info = {
        "pixels": pixels,
        "action": actions,
    }

    # --------------------------------------------------------
    # Encode all four states
    #
    # z0 z1 z2 z3
    # --------------------------------------------------------

    output = model.encode(info)

    emb = output["emb"]

    # B x 4 x 192

    act_emb = output["act_emb"]

    # B x 3 x 192

    # --------------------------------------------------------
    # Context
    #
    # z0 z1 z2
    # --------------------------------------------------------

    context_emb = emb[:, :HISTORY_SIZE]

    context_actions = act_emb[:, :HISTORY_SIZE]

    # --------------------------------------------------------
    # Targets
    #
    # z1 z2 z3
    #
    # Exactly the same shift used by original LeWM for
    # num_preds = 1.
    # --------------------------------------------------------

    target_emb = emb[:, 1:]

    # --------------------------------------------------------
    # Predict
    #
    # z0 + a0 -> z1_hat
    # z1 + a1 -> z2_hat
    # z2 + a2 -> z3_hat
    # --------------------------------------------------------

    predicted_emb = model.predict(
        context_emb,
        context_actions,
    )

    # --------------------------------------------------------
    # Original LeWM prediction loss
    # --------------------------------------------------------

    pred_loss = (
        predicted_emb - target_emb
    ).pow(2).mean()

    # --------------------------------------------------------
    # Original LeWM SIGReg
    #
    # Expected shape:
    # T x B x D
    # --------------------------------------------------------

    sigreg_loss = sigreg(
        emb.transpose(0, 1)
    )

    loss = (
        pred_loss
        +
        SIGREG_WEIGHT * sigreg_loss
    )

    return loss, pred_loss, sigreg_loss


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    sigreg,
    loader,
    device,
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

            (
                loss,
                pred_loss,
                sigreg_loss,
            ) = compute_loss(
                model,
                sigreg,
                batch,
                device,
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


# ============================================================
# Main training
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("Visual MiniGrid DoorKey LeWM training")
    print("=" * 70)

    print()
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ========================================================
    # Collect visual MiniGrid experience
    # ========================================================

    print()
    print(
        f"Collecting {NUM_TRANSITIONS} "
        f"visual transitions from {ENV_ID}..."
    )

    env = GymnasiumEnvironment(
        env_id=ENV_ID,
        observation_mode="pixels"
    )

    transitions = collect_transitions(
        env=env,
        num_steps=NUM_TRANSITIONS,
        seed=SEED,
        frame_skip=FRAME_SKIP,
    )

    env.close()

    episode_count = len(
        set(t.episode_id for t in transitions)
    )

    print(
        f"Collected {len(transitions)} transitions "
        f"from {episode_count} episodes."
    )

    # ========================================================
    # Episode-level train / validation split
    # ========================================================

    (
        train_transitions,
        val_transitions,
    ) = split_transitions_by_episode(
        transitions,
        train_fraction=TRAIN_FRACTION,
        seed=SEED,
    )

    train_sequences = TransitionSequenceDataset(
        transitions=train_transitions,
        history_size=HISTORY_SIZE,
    )

    val_sequences = TransitionSequenceDataset(
        transitions=val_transitions,
        history_size=HISTORY_SIZE,
    )

    train_dataset = LeWMPixelSequenceDataset(
        sequence_dataset=train_sequences,
        image_size=IMAGE_SIZE,
        action_encoding="one_hot",
        num_actions=NUM_ACTIONS,
    )

    sample = train_dataset[0]

    sample = train_dataset[0]

    print()
    print("Example MiniGrid training sample")
    print("=" * 60)

    print(
        "pixels:",
        sample["pixels"].shape,
    )

    print(
        "actions:",
        sample["action"].shape,
    )

    print(
        sample["action"]
    )

    val_dataset = LeWMPixelSequenceDataset(
        sequence_dataset=val_sequences,
        image_size=IMAGE_SIZE,
        action_encoding="one_hot",
        num_actions=NUM_ACTIONS,
    )

    print()
    print(
        "Training sequences:",
        len(train_dataset),
    )

    print(
        "Validation sequences:",
        len(val_dataset),
    )

    # ========================================================
    # Data loaders
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print("Creating LeWM...")

    model = build_lewm().to(device)

    sigreg = SIGReg(
        knots=17,
        num_proj=1024,
    ).to(device)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Current LeWM training uses a short linear warmup
    # followed by cosine decay.

    total_steps = (
        EPOCHS * len(train_loader)
    )

    warmup_steps = max(
        1,
        int(0.01 * total_steps),
    )

    def lr_schedule(step):

        if step < warmup_steps:
            return (
                step + 1
            ) / warmup_steps

        progress = (
            step - warmup_steps
        ) / max(
            1,
            total_steps - warmup_steps,
        )

        return 0.5 * (
            1.0
            +
            math.cos(
                math.pi * progress
            )
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_schedule,
    )

    # ========================================================
    # Checkpoint directory
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    checkpoint_dir = (
        project_root
        / "data"
        / "checkpoints"
        / "minigrid_doorkey_lewm"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    # ========================================================
    # Training loop
    # ========================================================

    print()
    print("=" * 70)
    print("Visual MiniGrid DoorKey LeWM training")
    print("=" * 70)

    global_step = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        running_pred = 0.0
        running_sigreg = 0.0

        num_batches = 0

        for batch in train_loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):

                (
                    loss,
                    pred_loss,
                    sigreg_loss,
                ) = compute_loss(
                    model,
                    sigreg,
                    batch,
                    device,
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_pred += pred_loss.item()
            running_sigreg += sigreg_loss.item()

            num_batches += 1
            global_step += 1

        # ----------------------------------------------------
        # Epoch metrics
        # ----------------------------------------------------

        train_loss = (
            running_loss / num_batches
        )

        train_pred = (
            running_pred / num_batches
        )

        train_sigreg = (
            running_sigreg / num_batches
        )

        val_metrics = validate(
            model,
            sigreg,
            val_loader,
            device,
        )

        world_metrics = evaluate_world_model(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"lr={current_lr:.2e} | "
            f"train loss={train_loss:.4f} | "
            f"train pred={train_pred:.4f} | "
            f"train sigreg={train_sigreg:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val pred={val_metrics['pred_loss']:.4f} | "
            f"val sigreg={val_metrics['sigreg_loss']:.4f}"
        )
        print(
            "    WM metrics | "
            f"final_pred={world_metrics['final_prediction_mse']:.4f} | "
            f"copy={world_metrics['copy_mse']:.4f} | "
            f"copy gain={100 * world_metrics['improvement_over_copy']:.1f}% | "
            f"shuffled={world_metrics['shuffled_action_mse']:.4f} | "
            f"action gain={100 * world_metrics['action_gain']:.1f}% | "
            f"beats shuffled={100 * world_metrics['correct_beats_shuffled']:.1f}% | "
            f"retrieval={100 * world_metrics['retrieval_top1']:.1f}% | "
            f"latent std={world_metrics['latent_feature_std']:.3f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics["loss"]
            < best_val_loss
        ):

            best_val_loss = (
                val_metrics["loss"]
            )

            checkpoint_path = (
                checkpoint_dir
                / "best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_loss":
                        best_val_loss,

                    "config": {
                        "history_size":
                            HISTORY_SIZE,

                        "image_size":
                            IMAGE_SIZE,

                        "embed_dim":
                            EMBED_DIM,

                        "sigreg_weight":
                            SIGREG_WEIGHT,
                    },
                },
                checkpoint_path,
            )

            print(
                f"  -> saved best model to "
                f"{checkpoint_path}"
            )

    print()
    print("=" * 70)
    print("Training finished")
    print("=" * 70)

    print(
        "Best validation loss:",
        best_val_loss,
    )


if __name__ == "__main__":
    main()