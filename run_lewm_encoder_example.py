import torch

from stable_pretraining.backbone.utils import vit_hf

from src.environments.gymnasium_env import GymnasiumEnvironment
from src.data_prep import (
    collect_transitions,
    TransitionSequenceDataset,
    LeWMPixelSequenceDataset,
)


def main():

    # -----------------------------------------------------
    # 1. Create visual CartPole environment
    # -----------------------------------------------------

    env = GymnasiumEnvironment(
        env_id="CartPole-v1",
        observation_mode="pixels_with_state",
    )

    # -----------------------------------------------------
    # 2. Collect some experience
    # -----------------------------------------------------

    transitions = collect_transitions(
        env=env,
        num_steps=1000,
        seed=42,
    )

    print(
        f"Collected {len(transitions)} transitions."
    )

    # -----------------------------------------------------
    # 3. Build temporal sequences
    # -----------------------------------------------------

    sequence_dataset = TransitionSequenceDataset(
        transitions=transitions,
        history_size=3,
    )

    dataset = LeWMPixelSequenceDataset(
        sequence_dataset=sequence_dataset,
        image_size=224,
    )

    print(
        f"Created {len(dataset)} LeWM sequences."
    )

    sample = dataset[0]

    print()
    print("LeWM-ready sample")
    print("=" * 60)

    print(
        "pixels:",
        sample["pixels"].shape,
        sample["pixels"].dtype,
    )

    print(
        "actions:",
        sample["action"].shape,
        sample["action"],
    )

    print(
        "hidden states:",
        sample["state"].shape,
    )

    # -----------------------------------------------------
    # 4. Create EXACT SAME encoder architecture as LeWM
    # -----------------------------------------------------

    encoder = vit_hf(
        size="tiny",
        patch_size=14,
        image_size=224,
        pretrained=False,
        use_mask_token=False,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    encoder = encoder.to(device)

    pixels = sample["pixels"].to(device)

    print()
    print("Encoding frames on:", device)

    # -----------------------------------------------------
    # 5. Run through ViT
    # -----------------------------------------------------

    with torch.no_grad():

        output = encoder(
            pixels,
            interpolate_pos_encoding=True,
        )

        # This is exactly what LeWM's JEPA.encode() uses.
        embeddings = output.last_hidden_state[:, 0]

    print()
    print("Encoder output:")
    print(
        "all tokens:",
        output.last_hidden_state.shape,
    )

    print(
        "CLS embeddings:",
        embeddings.shape,
    )

    print()

    print("Expected:")
    print("4 frames -> 4 latent vectors")
    print("each latent dimension = 192")

    print()

    print(
        "First latent vector, first 10 values:"
    )

    print(
        embeddings[0, :10]
            .detach()
            .cpu()
    )

    env.close()


if __name__ == "__main__":
    main()