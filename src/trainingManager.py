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
    loss_function = nn.BCEWithLogitsLoss()
    metric_names = ("iou", "precision", "recall")
    summary = []
    epochs = 2
    
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
        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_metrics = build_metrics()
        val_metrics = build_metrics()
        training_loader, validation_loader = get_training_data(val_split=0.2, orientation=orientation)
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
                loss = loss_function(predictions, labels)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
                train_batches += 1
                probs = torch.sigmoid(predictions.detach())
                targets = labels.detach().bool()
                train_metrics.update(probs, targets)
                current_iou = train_metrics.compute()["iou"].item()
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{current_iou:.4f}"})
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
                    loss = loss_function(predictions, labels)
                    val_loss += loss.item()
                    val_batches += 1
                    probs = torch.sigmoid(predictions)
                    targets = labels.bool()
                    val_metrics.update(probs, targets)
                    current_iou = val_metrics.compute()["iou"].item()
                    val_pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{current_iou:.4f}"})
            val_loss /= max(1, len(validation_loader))
            epoch_val_metrics = (
                {name: val_metrics.compute()[name].item() for name in metric_names}
                if val_batches
                else {name: float("nan") for name in metric_names}
            )
            final_val_loss = val_loss
            final_val_metrics = epoch_val_metrics

            # Save model if it has the best validation IoU so far
            current_val_iou = epoch_val_metrics.get("iou", 0.0)
            if current_val_iou > best_val_iou:
                best_val_iou = current_val_iou
                model_path = model_dir / f"best_{orientation}_{timestamp}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
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
            print(
                f"{orientation} [{epoch:02d}] train_bce={train_loss:.4f} | val_bce={val_loss:.4f} | "
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