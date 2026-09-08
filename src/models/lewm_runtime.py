import torch
from torch import nn
from omegaconf import DictConfig
from stable_pretraining.backbone.utils import vit_hf

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg


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

    action_encoder = Embedder(
        input_dim=action_input_dim,
        emb_dim=cfg.embed_dim,
    )

    projector = MLP(
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.projector.hidden_dim,
        output_dim=cfg.embed_dim,
        norm_fn=nn.BatchNorm1d,
    )

    pred_proj = MLP(
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.projector.hidden_dim,
        output_dim=cfg.embed_dim,
        norm_fn=nn.BatchNorm1d,
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )

    sigreg = SIGReg(**dict(cfg.loss.sigreg.kwargs))
    return model, sigreg


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

    context_emb = emb[:, :history_size]
    context_actions = act_emb[:, :history_size]
    target_emb = emb[:, 1:]

    predicted_emb = model.predict(context_emb, context_actions)

    pred_loss = (predicted_emb - target_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss = pred_loss + sigreg_weight * sigreg_loss

    return loss, pred_loss, sigreg_loss
