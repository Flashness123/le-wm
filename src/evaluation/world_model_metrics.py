from typing import Callable, Optional

import torch
import torch.nn.functional as F


CounterfactualActionFn = Callable[[torch.Tensor], torch.Tensor]


def flip_binary_centered_actions(
    actions: torch.Tensor,
) -> torch.Tensor:
    """
    For binary actions represented as:

        -1 = action 0
        +1 = action 1

    return the opposite action.

    Useful for CartPole.
    """

    return -actions


@torch.no_grad()
def evaluate_world_model(
    model,
    dataloader,
    device,
    counterfactual_action_fn: Optional[
        CounterfactualActionFn
    ] = None,
) -> dict[str, float]:
    """
    Generic latent-world-model evaluation.

    Expected batch:

        pixels:
            (B, T+1, C, H, W)

        action:
            (B, T, action_dim)

    Evaluates:

        1. prediction error
        2. copy-state baseline
        3. improvement over copying
        4. shuffled-action baseline
        5. action-conditioning gain
        6. fraction where correct action beats shuffled action
        7. in-batch next-state retrieval accuracy
        8. latent feature standard deviation
        9. optional counterfactual-action baseline
    """

    model.eval()

    totals = {}
    total_samples = 0

    for batch in dataloader:

        pixels = batch["pixels"].to(
            device,
            non_blocking=True,
        )

        actions = batch["action"].to(
            device,
            non_blocking=True,
        )

        batch_size = pixels.shape[0]

        # --------------------------------------------------
        # Encode the real trajectory
        # --------------------------------------------------

        info = {
            "pixels": pixels,
            "action": actions,
        }

        encoded = model.encode(info)

        embeddings = encoded["emb"]
        action_embeddings = encoded["act_emb"]

        # embeddings:
        #
        # z0 z1 z2 z3
        #
        # shape:
        # B x (T+1) x D

        context_embeddings = embeddings[:, :-1]
        target_embeddings = embeddings[:, 1:]

        # --------------------------------------------------
        # Correct-action prediction
        # --------------------------------------------------

        predicted_embeddings = model.predict(
            context_embeddings,
            action_embeddings,
        )

        # All prediction steps
        prediction_mse = F.mse_loss(
            predicted_embeddings,
            target_embeddings,
        )

        # Last prediction only
        pred_final = predicted_embeddings[:, -1]
        target_final = target_embeddings[:, -1]

        final_prediction_mse = F.mse_loss(
            pred_final,
            target_final,
        )

        # --------------------------------------------------
        # COPY BASELINE
        #
        # Predict:
        #
        # z_(t+1) = z_t
        #
        # If JEPA cannot beat this, it may simply be
        # exploiting temporal similarity.
        # --------------------------------------------------

        copy_final = context_embeddings[:, -1]

        copy_mse = F.mse_loss(
            copy_final,
            target_final,
        )

        if copy_mse.item() > 0:
            improvement_over_copy = (
                1.0
                - final_prediction_mse.item()
                / copy_mse.item()
            )
        else:
            improvement_over_copy = 0.0

        # --------------------------------------------------
        # SHUFFLED-ACTION BASELINE
        #
        # Give states the actions from another trajectory.
        #
        # This is generic and does not assume anything about
        # action semantics.
        # --------------------------------------------------

        shuffled_action_mse = float("nan")
        action_gain = float("nan")
        correct_beats_shuffled = float("nan")

        if batch_size > 1:

            shuffled_actions = torch.roll(
                actions,
                shifts=1,
                dims=0,
            )

            shuffled_action_embeddings = (
                model.action_encoder(
                    shuffled_actions
                )
            )

            shuffled_predictions = model.predict(
                context_embeddings,
                shuffled_action_embeddings,
            )

            shuffled_final = (
                shuffled_predictions[:, -1]
            )

            shuffled_action_mse_tensor = (
                F.mse_loss(
                    shuffled_final,
                    target_final,
                )
            )

            shuffled_action_mse = (
                shuffled_action_mse_tensor.item()
            )

            if shuffled_action_mse > 0:
                action_gain = (
                    1.0
                    - final_prediction_mse.item()
                    / shuffled_action_mse
                )

            # Per-sample errors
            correct_error_per_sample = (
                (pred_final - target_final)
                .pow(2)
                .mean(dim=-1)
            )

            shuffled_error_per_sample = (
                (shuffled_final - target_final)
                .pow(2)
                .mean(dim=-1)
            )

            correct_beats_shuffled = (
                (
                    correct_error_per_sample
                    <
                    shuffled_error_per_sample
                )
                .float()
                .mean()
                .item()
            )

        # --------------------------------------------------
        # NEXT-STATE RETRIEVAL
        #
        # For each predicted future latent, find the nearest
        # actual target latent in the current batch.
        #
        # Correct answer should be the corresponding sample.
        # --------------------------------------------------

        retrieval_top1 = float("nan")

        if batch_size > 1:

            distances = torch.cdist(
                pred_final.float(),
                target_final.float(),
                p=2,
            )

            nearest_target = (
                distances.argmin(dim=1)
            )

            correct_target = torch.arange(
                batch_size,
                device=device,
            )

            retrieval_top1 = (
                (
                    nearest_target
                    == correct_target
                )
                .float()
                .mean()
                .item()
            )

        # --------------------------------------------------
        # LATENT COLLAPSE DIAGNOSTIC
        #
        # If this becomes close to zero, representations
        # may have collapsed.
        # --------------------------------------------------

        flattened_embeddings = (
            embeddings
            .float()
            .reshape(
                -1,
                embeddings.shape[-1],
            )
        )

        latent_feature_std = (
            flattened_embeddings
            .std(dim=0)
            .mean()
            .item()
        )

        # --------------------------------------------------
        # OPTIONAL ENVIRONMENT-SPECIFIC COUNTERFACTUAL
        #
        # CartPole:
        #
        # LEFT  -> RIGHT
        # RIGHT -> LEFT
        # --------------------------------------------------

        counterfactual_mse = float("nan")
        counterfactual_gain = float("nan")
        correct_beats_counterfactual = float("nan")

        if counterfactual_action_fn is not None:

            counterfactual_actions = (
                counterfactual_action_fn(actions)
            )

            counterfactual_action_embeddings = (
                model.action_encoder(
                    counterfactual_actions
                )
            )

            counterfactual_predictions = (
                model.predict(
                    context_embeddings,
                    counterfactual_action_embeddings,
                )
            )

            counterfactual_final = (
                counterfactual_predictions[:, -1]
            )

            counterfactual_mse_tensor = (
                F.mse_loss(
                    counterfactual_final,
                    target_final,
                )
            )

            counterfactual_mse = (
                counterfactual_mse_tensor.item()
            )

            if counterfactual_mse > 0:
                counterfactual_gain = (
                    1.0
                    - final_prediction_mse.item()
                    / counterfactual_mse
                )

            correct_error_per_sample = (
                (pred_final - target_final)
                .pow(2)
                .mean(dim=-1)
            )

            counterfactual_error_per_sample = (
                (
                    counterfactual_final
                    - target_final
                )
                .pow(2)
                .mean(dim=-1)
            )

            correct_beats_counterfactual = (
                (
                    correct_error_per_sample
                    <
                    counterfactual_error_per_sample
                )
                .float()
                .mean()
                .item()
            )

        # --------------------------------------------------
        # Aggregate
        # --------------------------------------------------

        batch_metrics = {
            "prediction_mse":
                prediction_mse.item(),

            "final_prediction_mse":
                final_prediction_mse.item(),

            "copy_mse":
                copy_mse.item(),

            "shuffled_action_mse":
                shuffled_action_mse,

            "correct_beats_shuffled":
                correct_beats_shuffled,

            "retrieval_top1":
                retrieval_top1,

            "latent_feature_std":
                latent_feature_std,

            "counterfactual_mse":
                counterfactual_mse,

            "correct_beats_counterfactual":
                correct_beats_counterfactual,
        }

        for key, value in batch_metrics.items():

            if value != value:  # NaN
                continue

            totals[key] = (
                totals.get(key, 0.0)
                + value * batch_size
            )

        total_samples += batch_size

    metrics = {
        key: value / total_samples
        for key, value in totals.items()
    }

    if metrics["copy_mse"] > 0:
        metrics["improvement_over_copy"] = (
            1.0
            - metrics["final_prediction_mse"]
            / metrics["copy_mse"]
        )

    if metrics["shuffled_action_mse"] > 0:
        metrics["action_gain"] = (
            1.0
            - metrics["final_prediction_mse"]
            / metrics["shuffled_action_mse"]
        )

    if (
        "counterfactual_mse" in metrics
        and metrics["counterfactual_mse"] > 0
    ):
        metrics["counterfactual_gain"] = (
            1.0
            - metrics["final_prediction_mse"]
            / metrics["counterfactual_mse"]
        )

    return metrics