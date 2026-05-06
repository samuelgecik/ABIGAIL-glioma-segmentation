import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall

from src.data_manager import get_training_data_3d
from src.model3d import UNet3D


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for binary segmentation."""

    def __init__(self, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE component
        bce_loss = self.bce(logits, targets)

        # Dice component (operates on probabilities)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def _prepare_batch_3d(images, labels):
    """Pad spatial dimensions to be divisible by 16 for 3D U-Net."""
    _, _, d, h, w = images.shape
    pad_d = (16 - d % 16) % 16
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_d or pad_h or pad_w:
        # F.pad order: (w_left, w_right, h_left, h_right, d_left, d_right)
        pad = (0, pad_w, 0, pad_h, 0, pad_d)
        images = F.pad(images, pad, mode="reflect")
        labels = F.pad(labels, pad, mode="constant", value=0)
    return images, labels


def _write_logs(log_path: Path, records: list) -> None:
    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D UNet for volumetric brain tumor segmentation.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience (default: 30)")
    parser.add_argument('--val-every', type=int, default=10, help="Validate every N epochs (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: UNet3D (full-volume, batch_size=4, gradient checkpointing)")

    metric_names = ("iou", "precision", "recall")

    # Directories
    run_dir = Path("training_logs")
    run_dir.mkdir(exist_ok=True)
    model_dir = Path("saved_models")
    model_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = run_dir / f"unet3d_run_{timestamp}.json"

    # Data
    print("\nLoading 3D training data...")
    train_loader, val_loader = get_training_data_3d(preload=True, augment=True)

    # Model
    model = UNet3D(in_channels=1, out_classes=1, base_filters=32, use_checkpoint=True).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Loss, optimizer, scheduler
    loss_fn = DiceBCELoss(dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)

    # AMP
    scaler = torch.amp.GradScaler()

    # Metrics
    def build_metrics():
        return MetricCollection({
            "iou": BinaryJaccardIndex(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }).to(device)

    train_metrics = build_metrics()
    val_metrics = build_metrics()

    log_records = []
    best_val_iou = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # ---- Training ----
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_metrics.reset()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", unit="batch")
        for images, labels in train_pbar:
            images, labels = _prepare_batch_3d(images, labels)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                predictions = model(images)
                loss = loss_fn(predictions, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

            with torch.no_grad():
                probs = torch.sigmoid(predictions.detach())
                train_metrics.update(probs, labels.bool())

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        train_loss = train_loss_sum / max(1, train_batches)
        epoch_train_metrics = {
            name: train_metrics.compute()[name].item() for name in metric_names
        } if train_batches else {name: float("nan") for name in metric_names}

        # ---- Validation (every val_every epochs) ----
        epoch_val_loss = float("nan")
        epoch_val_metrics = {name: float("nan") for name in metric_names}

        if epoch % args.val_every == 0 or epoch == 1:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            val_metrics.reset()

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]  ", unit="batch")
            with torch.no_grad():
                for images, labels in val_pbar:
                    images, labels = _prepare_batch_3d(images, labels)
                    images = images.to(device)
                    labels = labels.to(device)

                    with torch.amp.autocast('cuda'):
                        predictions = model(images)
                        loss = loss_fn(predictions, labels)

                    val_loss_sum += loss.item()
                    val_batches += 1
                    probs = torch.sigmoid(predictions)
                    val_metrics.update(probs, labels.bool())

                    val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_val_loss = val_loss_sum / max(1, val_batches)
            epoch_val_metrics = {
                name: val_metrics.compute()[name].item() for name in metric_names
            } if val_batches else {name: float("nan") for name in metric_names}

            current_val_iou = epoch_val_metrics.get("iou", 0.0)

            # Save best model
            if current_val_iou > best_val_iou:
                best_val_iou = current_val_iou
                epochs_without_improvement = 0
                model_path = model_dir / f"best_unet3d_{timestamp}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_iou': current_val_iou,
                    'val_metrics': epoch_val_metrics,
                    'model_arch': 'unet3d',
                }, model_path)
                print(f"  -> Saved best model (IoU: {current_val_iou:.4f}) to {model_path}")
            else:
                epochs_without_improvement += args.val_every

        # ---- Logging ----
        train_str = " ".join(f"train_{n}={epoch_train_metrics[n]:.4f}" for n in metric_names)
        val_str = " ".join(f"val_{n}={epoch_val_metrics[n]:.4f}" for n in metric_names)
        lr = optimizer.param_groups[0]['lr']
        print(f"[{epoch:03d}] LR={lr:.6f} | train_loss={train_loss:.4f} | val_loss={epoch_val_loss:.4f} | {train_str} | {val_str}")

        log_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": epoch_val_loss,
            "train_metrics": epoch_train_metrics,
            "val_metrics": epoch_val_metrics,
        })
        _write_logs(log_path, log_records)

        # ---- Early stopping ----
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    print(f"\nTraining complete. Best Val IoU: {best_val_iou:.4f}")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
