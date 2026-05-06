import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score

from src.data_manager import get_training_data
from src.model import UNet, DeepLabV3, NestedUNet

VALID_ARCHITECTURES = ('unet', 'deeplabv3', 'nestedunet')


def parse_args():
    parser = argparse.ArgumentParser(description="Run validation inference on trained models.")
    parser.add_argument(
        '--model-arch',
        type=str,
        default='unet',
        choices=VALID_ARCHITECTURES,
        help="Model architecture to evaluate (default: unet)",
    )
    return parser.parse_args()


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


def load_model_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load a trained model from checkpoint."""
    # Load checkpoint to check model architecture
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model architecture (default to UNet for backwards compatibility)
    model_arch = checkpoint.get('model_arch', 'unet')
    
    # Initialize the appropriate model
    if model_arch == 'deeplabv3':
        model = DeepLabV3(in_channels=1, out_classes=1).to(device)
    elif model_arch == 'nestedunet':
        model = NestedUNet(in_ch=1, out_ch=1).to(device)
    else:
        model = UNet(in_channels=1, out_classes=1, up_sample_mode='conv_transpose').to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {model_arch.upper()} checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val IoU: {checkpoint['val_iou']:.4f}")
    
    return model


def evaluate_model(
    model,
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
    args = parse_args()
    model_arch = args.model_arch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model architecture: {model_arch}")

    # Directory containing saved models
    model_dir = Path("saved_models")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Build glob pattern based on architecture.
    # UNet files: best_{orientation}_{timestamp}.pt  (no arch prefix)
    # Others:     best_{arch}_{orientation}_{timestamp}.pt
    if model_arch == 'unet':
        # Match unet files but exclude files that have a known arch prefix
        glob_pattern = "best_*.pt"
        arch_prefixes = tuple(f"best_{a}_" for a in VALID_ARCHITECTURES if a != 'unet')
    else:
        glob_pattern = f"best_{model_arch}_*.pt"
        arch_prefixes = None  # no filtering needed

    model_files = list(model_dir.glob(glob_pattern))
    # For unet, exclude files belonging to other architectures
    if arch_prefixes:
        model_files = [f for f in model_files if not f.name.startswith(arch_prefixes)]

    if not model_files:
        raise FileNotFoundError(f"No {model_arch} model files found in {model_dir}")

    # Parse orientation and timestamp from filenames.
    # UNet:   best_{orientation}_{timestamp}.pt  → prefix length 1 (skip 'best')
    # Others: best_{arch}_{orientation}_{timestamp}.pt → prefix length 2 (skip 'best', arch)
    from collections import defaultdict
    timestamp_groups = defaultdict(list)
    prefix_parts = 1 if model_arch == 'unet' else 2
    for model_file in model_files:
        parts = model_file.stem.split("_")
        if len(parts) >= prefix_parts + 2:
            orientation = parts[prefix_parts]
            timestamp = "_".join(parts[prefix_parts + 1:])
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
        raise FileNotFoundError(
            f"Could not find a complete set of {model_arch} models (axial, coronal, sagittal)"
        )
    
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
        _, validation_loader = get_training_data(orientation=orientation, preload=False)
        
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
