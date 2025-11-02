import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from brats_mvp.dataManager import get_training_data
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryDice, BinaryJaccardIndex, BinaryPrecision, BinaryRecall


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
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    model = model.to(device)

    training_loader, validation_loader = get_training_data(val_split=0.2)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    metric_names = ("dice", "iou", "precision", "recall")

    def build_metrics() -> MetricCollection:
        collection = MetricCollection(
            {
                "dice": BinaryDice(),
                "iou": BinaryJaccardIndex(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
            }
        )
        return collection.to(device)

    train_metrics = build_metrics()
    val_metrics = build_metrics()

    epochs = 2

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        train_metrics.reset()
        train_batches = 0

        train_pbar = tqdm(training_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch")
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

            current_dice = train_metrics.compute()["dice"].item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{current_dice:.4f}"})

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

        val_pbar = tqdm(validation_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", unit="batch")
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

                current_dice = val_metrics.compute()["dice"].item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{current_dice:.4f}"})
        val_loss /= max(1, len(validation_loader))

        epoch_val_metrics = (
            {name: val_metrics.compute()[name].item() for name in metric_names}
            if val_batches
            else {name: float("nan") for name in metric_names}
        )

        train_metric_str = " ".join(
            f"train_{name}={epoch_train_metrics[name]:.4f}" for name in metric_names
        )
        val_metric_str = " ".join(
            f"val_{name}={epoch_val_metrics[name]:.4f}" for name in metric_names
        )

        print(
            f"[{epoch:02d}] train_bce={train_loss:.4f} | val_bce={val_loss:.4f} | "
            f"{train_metric_str} | {val_metric_str}"
        )


if __name__ == "__main__":
    main()