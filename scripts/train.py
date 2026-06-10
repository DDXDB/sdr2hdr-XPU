from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sdr2hdr.dataset import HDRSDRPairDataset
from sdr2hdr.model import EnhancementUNet


def resolve_training_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def total_variation_loss(tensor: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).mean()
    dy = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).mean()
    return dx + dy


def _compute_luma(frame: torch.Tensor) -> torch.Tensor:
    return frame[:, 0:1] * 0.2627 + frame[:, 1:2] * 0.6780 + frame[:, 2:3] * 0.0593


def _compute_heuristic_base_maps(sdr_linear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sdr_luma = torch.clamp(_compute_luma(sdr_linear), 0.0, 1.0)
    smooth = torch.nn.functional.avg_pool2d(sdr_luma, 11, stride=1, padding=5)
    detail = torch.clamp(sdr_luma - smooth, -0.2, 0.2)
    base_exp = torch.clamp((sdr_luma - 0.55) / 0.45, 0.0, 1.0)
    base_con = torch.clamp(torch.abs(detail) * 6.0, 0.0, 1.0)
    return base_exp, base_con


def compute_loss(
    pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_maps = batch["target_maps"]
    clip_mask = batch["clip_mask"]
    near_white_mask = batch["near_white_mask"]
    shadow_mask = batch["shadow_mask"]
    memory_color_mask = batch["memory_color_mask"]
    region_weight = batch["region_weight"]
    sdr_linear = batch["sdr_linear"]
    confidence = 1.0 - clip_mask * 0.8
    pred_exp, pred_con, pred_pro = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    tgt_exp, tgt_con, tgt_pro = target_maps[:, 0:1], target_maps[:, 1:2], target_maps[:, 2:3]
    base_exp, base_con = _compute_heuristic_base_maps(sdr_linear)
    pred_exp_abs = torch.clamp(base_exp + pred_exp, 0.0, 1.0)
    tgt_exp_abs = torch.clamp(base_exp + tgt_exp, 0.0, 1.0)
    pred_con_abs = torch.clamp(base_con + pred_con, 0.0, 1.0)
    tgt_con_abs = torch.clamp(base_con + tgt_con, 0.0, 1.0)
    sdr_luma = torch.clamp(_compute_luma(sdr_linear), 0.0, 1.0)
    pred_tone = sdr_luma + pred_exp_abs * (1.0 - sdr_luma)
    tgt_tone = sdr_luma + tgt_exp_abs * (1.0 - sdr_luma)
    protected_weight = 1.0 + torch.clamp(tgt_pro, min=0.0) * 1.5
    overdrive = torch.clamp(pred_exp - tgt_exp, min=0.0)
    weighted_confidence = confidence * region_weight
    l_exp = (weighted_confidence * protected_weight * torch.abs(pred_exp - tgt_exp)).mean()
    l_exp = l_exp + (protected_weight * overdrive).mean() * 0.5
    l_con = (weighted_confidence * (1.0 + torch.clamp(tgt_pro, min=0.0)) * torch.abs(pred_con - tgt_con)).mean()
    l_pro = torch.abs(pred_pro - tgt_pro).mean()
    l_tone = torch.abs(pred_tone - tgt_tone).mean()
    l_memory = (memory_color_mask * torch.abs(pred_exp_abs - tgt_exp_abs)).mean()
    l_memory = l_memory + 0.35 * (memory_color_mask * torch.abs(pred_con_abs - tgt_con_abs)).mean()
    shadow_overdrive = torch.clamp(pred_exp_abs - tgt_exp_abs, min=0.0)
    l_shadow = (shadow_mask * shadow_overdrive).mean() + 0.2 * (shadow_mask * torch.clamp(pred_con_abs - tgt_con_abs, min=0.0)).mean()
    l_rolloff = (near_white_mask * torch.clamp(pred_exp_abs - tgt_exp_abs, min=0.0)).mean()
    l_tv = total_variation_loss(pred_exp) * 0.01
    total = 1.0 * l_exp + 0.5 * l_con + 0.75 * l_pro + 0.6 * l_tone + 0.45 * l_memory + 0.35 * l_shadow + 0.4 * l_rolloff + l_tv
    return total, {
        "exp": l_exp,
        "con": l_con,
        "pro": l_pro,
        "tone": l_tone,
        "memory": l_memory,
        "shadow": l_shadow,
        "rolloff": l_rolloff,
        "tv": l_tv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train enhancement map estimator.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    train_source = HDRSDRPairDataset(args.data_dir, patch_size=args.patch_size, training=True)
    val_source = HDRSDRPairDataset(args.data_dir, patch_size=args.patch_size, training=False)
    indices = list(range(len(train_source)))
    val_size = max(1, len(indices) // 10)
    train_indices = indices[:-val_size] or indices
    val_indices = indices[-val_size:]
    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = resolve_training_device(args.device)
    model = EnhancementUNet().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    warmup = LinearLR(optimizer, start_factor=0.2, total_iters=3)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 3))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train {epoch+1}/{args.epochs}"):
            sdr_linear = batch["sdr_linear"].to(device)
            device_batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(sdr_linear)
                loss, _ = compute_loss(pred, device_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu())
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val {epoch+1}/{args.epochs}"):
                sdr_linear = batch["sdr_linear"].to(device)
                device_batch = {key: value.to(device) for key, value in batch.items()}
                pred = model(sdr_linear)
                loss, losses = compute_loss(pred, device_batch)
                val_loss += float(loss.detach().cpu())
        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        for name, value in losses.items():
            writer.add_scalar(f"loss_component/{name}", float(value.detach().cpu()), epoch)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, output_dir / "best.pt")

    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
