import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.dataManager import get_training_data
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall


def calculate_pos_weight(data_loader, device, sample_batches=50):
    """
    Calculate positive weight for BCEWithLogitsLoss based on class distribution.
    Samples a subset of batches to estimate the ratio.
    
    Returns:
        pos_weight: Tensor with weight for positive class
    """
    print("Calculating class distribution for loss weighting...")
    total_positive = 0
    total_negative = 0
    
    for i, (_, labels) in enumerate(data_loader):
        if i >= sample_batches:
            break
        labels = labels.to(device)
        total_positive += labels.sum().item()
        total_negative += (1 - labels).sum().item()
    
    if total_positive == 0:
        print("WARNING: No positive samples found in subset!")
        return torch.tensor([1.0]).to(device)
    
    pos_weight_value = total_negative / total_positive
    print(f"  Positive pixels: {total_positive:,.0f}")
    print(f"  Negative pixels: {total_negative:,.0f}")
    print(f"  Calculated pos_weight: {pos_weight_value:.2f}")
    
    return torch.tensor([pos_weight_value]).to(device)


def _prepare_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match hub U-Net expectations (3 channels, spatial dims divisible by 16)."""
    if images.size(1) != 3:
        images = images.repeat(1, 3, 1, 1)

    _, _, height, width = images.shape
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        pad = (0, pad_w, 0, pad_h)
        images = F.pad(images, pad, mode="reflect")
        labels = F.pad(labels, pad, mode="constant", value=0)

    return images, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    metric_names = ("iou", "precision", "recall")
    summary = []
    epochs = 20
    
    # Create directories for saving models and logs
    run_dir = Path("training_logs")
    run_dir.mkdir(exist_ok=True)
    model_dir = Path("saved_models")
    model_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    def build_metrics() -> MetricCollection:
        collection = MetricCollection(
            {
                "iou": BinaryJaccardIndex(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
            }
        )
        return collection.to(device)

    log_records = []
    for orientation in ("axial", "coronal", "sagittal"):
        print(f"\n{'='*60}")
        print(f"Training orientation: {orientation.upper()}")
        print(f"{'='*60}")
        
        # Load data first to calculate pos_weight
        training_loader, validation_loader = get_training_data(val_split=0.2, orientation=orientation)
        
        pos_weight = calculate_pos_weight(training_loader, device, sample_batches=50)
        loss_function_oriented = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # ReduceLROnPlateau will lower LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        
        train_metrics = build_metrics()
        val_metrics = build_metrics()
        final_val_loss = float("nan")
        final_val_metrics = {name: float("nan") for name in metric_names}
        best_val_iou = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            training_loss = 0.0
            train_metrics.reset()
            train_batches = 0
            train_pbar = tqdm(training_loader, desc=f"Orientation {orientation} Epoch {epoch}/{epochs} [Train]", unit="batch")
            for images, labels in train_pbar:
                images, labels = _prepare_batch(images, labels)
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(images)
                loss = loss_function_oriented(predictions, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                training_loss += loss.item()
                train_batches += 1
                probs = torch.sigmoid(predictions.detach())
                targets = labels.detach().bool()
                train_metrics.update(probs, targets)
                
                if train_batches % 100 == 0:  # Show metrics every 100 batches
                    current_metrics = train_metrics.compute()
                    train_pbar.set_postfix({
                        "loss": f"{loss.item():.4f}", 
                        "iou": f"{current_metrics['iou'].item():.4f}"
                    })
                else:
                    train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    
            train_loss = training_loss / max(1, len(training_loader))
            epoch_train_metrics = (
                {name: train_metrics.compute()[name].item() for name in metric_names}
                if train_batches
                else {name: float("nan") for name in metric_names}
            )

            model.eval()
            val_loss = 0.0
            val_metrics.reset()
            val_batches = 0
            val_pbar = tqdm(validation_loader, desc=f"{orientation} Epoch {epoch}/{epochs} [Val]  ", unit="batch")
            with torch.no_grad():
                for images, labels in val_pbar:
                    images, labels = _prepare_batch(images, labels)
                    images = images.to(device)
                    labels = labels.to(device)
                    predictions = model(images)
                    loss = loss_function_oriented(predictions, labels)
                    val_loss += loss.item()
                    val_batches += 1
                    probs = torch.sigmoid(predictions)
                    targets = labels.bool()
                    val_metrics.update(probs, targets)
                    
                    if val_batches % 100 == 0:
                        current_metrics = val_metrics.compute()
                        val_pbar.set_postfix({
                            "loss": f"{loss.item():.4f}", 
                            "iou": f"{current_metrics['iou'].item():.4f}"
                        })
                    else:
                        val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                        
            val_loss /= max(1, len(validation_loader))
            epoch_val_metrics = (
                {name: val_metrics.compute()[name].item() for name in metric_names}
                if val_batches
                else {name: float("nan") for name in metric_names}
            )
            final_val_loss = val_loss
            final_val_metrics = epoch_val_metrics
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  → Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

            # Save model if it has the best validation IoU so far
            current_val_iou = epoch_val_metrics.get("iou", 0.0)
            if current_val_iou > best_val_iou:
                best_val_iou = current_val_iou
                model_path = model_dir / f"best_{orientation}_{timestamp}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_iou': current_val_iou,
                    'val_metrics': epoch_val_metrics,
                }, model_path)
                print(f"  → Saved best model for {orientation} (IoU: {current_val_iou:.4f}) to {model_path}")

            train_metric_str = " ".join(
                f"train_{name}={epoch_train_metrics[name]:.4f}" for name in metric_names
            )
            val_metric_str = " ".join(
                f"val_{name}={epoch_val_metrics[name]:.4f}" for name in metric_names
            )
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"{orientation} [{epoch:02d}] LR={current_lr:.6f} | train_bce={train_loss:.4f} | val_bce={val_loss:.4f} | "
                f"{train_metric_str} | {val_metric_str}"
            )
            log_records.append(
                {
                    "orientation": orientation,
                    "epoch": epoch,
                    "train_bce": train_loss,
                    "val_bce": val_loss,
                    "train_metrics": epoch_train_metrics,
                    "val_metrics": epoch_val_metrics,
                }
            )

        summary.append((orientation, final_val_loss, final_val_metrics))
        print(f"  → Best IoU for {orientation}: {best_val_iou:.4f}")

    if summary:
        avg_loss = sum(item[1] for item in summary) / len(summary)
        avg_metrics = {
            name: sum(item[2][name] for item in summary) / len(summary)
            for name in metric_names
        }
        avg_metric_str = " ".join(f"avg_{name}={avg_metrics[name]:.4f}" for name in metric_names)
        print(f"mean_val_bce={avg_loss:.4f} | {avg_metric_str}")
        log_records.append(
            {
                "orientation": "summary",
                "epoch": None,
                "val_bce": avg_loss,
                "val_metrics": avg_metrics,
            }
        )

    log_path = run_dir / f"three_axis_run_{timestamp}.json"
    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(log_records, fp, indent=2)
    print(f"Saved log to {log_path}")


if __name__ == "__main__":
    main()