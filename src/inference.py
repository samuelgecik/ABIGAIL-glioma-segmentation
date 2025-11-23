import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score

from src.dataManager import get_training_data
from src.model import UNet


def _prepare_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad spatial dimensions to be divisible by 16 for U-Net."""
    _, _, height, width = images.shape
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        pad = (0, pad_w, 0, pad_h)
        images = F.pad(images, pad, mode="reflect")
        labels = F.pad(labels, pad, mode="constant", value=0)
    return images, labels


def load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> UNet:
    """Load a trained model from checkpoint."""
    model = UNet(in_channels=1, out_classes=1, up_sample_mode='conv_transpose').to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val IoU: {checkpoint['val_iou']:.4f}")
    
    return model


def evaluate_model(
    model: UNet,
    validation_loader,
    device: torch.device,
    orientation: str
) -> Dict[str, float]:
    """Evaluate model on validation set and return metrics."""
    metrics = MetricCollection({
        "iou": BinaryJaccardIndex(),
        "precision": BinaryPrecision(),
        "recall": BinaryRecall(),
        "f1": BinaryF1Score(),
    }).to(device)
    
    model.eval()
    metrics.reset()
    
    print(f"\nEvaluating {orientation} model on validation set...")
    pbar = tqdm(validation_loader, desc=f"{orientation} Inference", unit="batch")
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = _prepare_batch(images, labels)
            images = images.to(device)
            labels = labels.to(device)
            
            predictions = model(images)
            probs = torch.sigmoid(predictions)
            targets = labels.bool()
            
            metrics.update(probs, targets)
            
            # Show running metrics
            current_metrics = metrics.compute()
            pbar.set_postfix({
                "iou": f"{current_metrics['iou'].item():.4f}",
                "f1": f"{current_metrics['f1'].item():.4f}"
            })
    
    final_metrics = metrics.compute()
    result = {name: final_metrics[name].item() for name in ["iou", "precision", "recall", "f1"]}
    
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Directory containing saved models
    model_dir = Path("saved_models")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find the most recent timestamp for the three models
    # Expected naming: best_{orientation}_{timestamp}.pt
    model_files = list(model_dir.glob("best_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    # Group by timestamp to find the latest complete set
    from collections import defaultdict
    timestamp_groups = defaultdict(list)
    for model_file in model_files:
        parts = model_file.stem.split("_")  # e.g., ['best', 'axial', '20251119-192024']
        if len(parts) >= 3:
            orientation = parts[1]
            timestamp = "_".join(parts[2:])
            timestamp_groups[timestamp].append((orientation, model_file))
    
    # Find the most recent complete set (all three orientations)
    latest_timestamp = None
    latest_models = None
    for timestamp, models in sorted(timestamp_groups.items(), reverse=True):
        orientations = {ori for ori, _ in models}
        if {"axial", "coronal", "sagittal"}.issubset(orientations):
            latest_timestamp = timestamp
            latest_models = {ori: path for ori, path in models}
            break
    
    if latest_models is None:
        raise FileNotFoundError("Could not find a complete set of models (axial, coronal, sagittal)")
    
    print(f"\nFound complete model set with timestamp: {latest_timestamp}")
    print(f"Models:")
    for ori in ["axial", "coronal", "sagittal"]:
        print(f"  {ori}: {latest_models[ori]}")
    
    # Run inference for each orientation
    results = {}
    
    for orientation in ["axial", "coronal", "sagittal"]:
        print(f"\n{'='*60}")
        print(f"Processing {orientation.upper()} orientation")
        print(f"{'='*60}")
        
        # Load validation data for this orientation (no preload needed for inference)
        _, validation_loader = get_training_data(val_split=0.2, orientation=orientation, preload=False)
        
        # Load model
        model = load_model_checkpoint(latest_models[orientation], device)
        
        # Evaluate
        metrics = evaluate_model(model, validation_loader, device, orientation)
        results[orientation] = metrics
        
        # Print results
        print(f"\n{orientation.upper()} Results:")
        print(f"  IoU (Dice): {metrics['iou']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1 Score:   {metrics['f1']:.4f}")
    
    # Compute average across orientations
    print(f"\n{'='*60}")
    print("SUMMARY - Average across all orientations")
    print(f"{'='*60}")
    avg_metrics = {
        metric: sum(results[ori][metric] for ori in ["axial", "coronal", "sagittal"]) / 3
        for metric in ["iou", "precision", "recall", "f1"]
    }
    print(f"  Average IoU:       {avg_metrics['iou']:.4f}")
    print(f"  Average Precision: {avg_metrics['precision']:.4f}")
    print(f"  Average Recall:    {avg_metrics['recall']:.4f}")
    print(f"  Average F1 Score:  {avg_metrics['f1']:.4f}")
    
    # Save results
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    timestamp_now = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"inference_{latest_timestamp}_{timestamp_now}.json"
    
    output_data = {
        "model_timestamp": latest_timestamp,
        "inference_timestamp": timestamp_now,
        "device": str(device),
        "results_by_orientation": results,
        "average_metrics": avg_metrics,
    }
    
    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
