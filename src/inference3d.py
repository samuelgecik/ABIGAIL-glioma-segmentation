"""
Full-volume 3D inference and evaluation for UNet3D.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecision, BinaryRecall, BinaryF1Score

from src.model3d import UNet3D
from src.utils import binarize_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D volumetric inference on trained UNet3D.")
    return parser.parse_args()


def _pad_volume(volume: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Pad a [B,1,D,H,W] tensor so spatial dims are divisible by 16. Returns padded tensor and original (D,H,W)."""
    _, _, d, h, w = volume.shape
    pad_d = (16 - d % 16) % 16
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_d or pad_h or pad_w:
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d), mode="reflect")
    return volume, (d, h, w)


def _unpad_volume(volume: torch.Tensor, original_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Remove padding to restore original spatial dimensions."""
    d, h, w = original_shape
    return volume[:, :, :d, :h, :w]


def load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> UNet3D:
    """Load a trained UNet3D from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet3D(in_channels=1, out_classes=1, base_filters=32, use_checkpoint=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded UNet3D checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val IoU: {checkpoint['val_iou']:.4f}")

    return model


def predict_single_case_3d(
    case_id: str,
    model_path: str,
    data_dir: str,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run 3D inference on a single case. Returns (prediction_volume, probability_volume).
    Both are in canonical orientation (D, H, W).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find image file
    case_dir = Path(data_dir) / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    candidates = [
        case_dir / f"{case_id}-t2f.nii.gz",
        case_dir / f"{case_id}-flair.nii.gz",
    ]
    img_path = next((str(p) for p in candidates if p.exists()), None)
    if img_path is None:
        raise FileNotFoundError(f"No T2-FLAIR image found for {case_id}")

    # Load volume
    raw_volume = nib.load(img_path).get_fdata(dtype=np.float32)

    # Volume-level z-score normalization (matching training)
    brain_mask = raw_volume > 0
    image = raw_volume.copy()
    if brain_mask.any():
        brain_voxels = image[brain_mask]
        mean, std = brain_voxels.mean(), brain_voxels.std()
        if std > 0:
            image = (image - mean) / std
            image[~brain_mask] = 0.0

    # Load model
    model = load_model_checkpoint(Path(model_path), device)

    # Forward pass
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
    input_padded, orig_shape = _pad_volume(input_tensor)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits = model(input_padded)

    logits = _unpad_volume(logits, orig_shape)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (D, H, W)

    pred = (probs > threshold).astype(np.uint8)

    print(f"Inference complete. Prediction shape: {pred.shape}")
    return pred, probs


def evaluate_validation_3d(
    model: UNet3D,
    val_pairs: List[Tuple[str, str]],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation volumes. Returns aggregated metrics."""
    metrics = MetricCollection({
        "iou": BinaryJaccardIndex(),
        "precision": BinaryPrecision(),
        "recall": BinaryRecall(),
        "f1": BinaryF1Score(),
    }).to(device)

    model.eval()
    metrics.reset()

    print(f"\nEvaluating on {len(val_pairs)} validation volumes...")
    for img_path, lbl_path in tqdm(val_pairs, desc="3D Inference", unit="vol"):
        # Load and normalize
        raw = nib.load(img_path).get_fdata(dtype=np.float32)
        brain_mask = raw > 0
        image = raw.copy()
        if brain_mask.any():
            voxels = image[brain_mask]
            mean, std = voxels.mean(), voxels.std()
            if std > 0:
                image = (image - mean) / std
                image[~brain_mask] = 0.0

        # Load ground truth
        gt = nib.load(lbl_path).get_fdata(dtype=np.float32).astype(np.int16)
        gt_binary = binarize_mask(gt)

        # Forward pass
        input_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
        input_padded, orig_shape = _pad_volume(input_t)
        gt_t = torch.from_numpy(gt_binary.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits = model(input_padded)

        logits = _unpad_volume(logits, orig_shape)
        probs = torch.sigmoid(logits)
        metrics.update(probs, gt_t.bool())

    result = metrics.compute()
    return {name: result[name].item() for name in ["iou", "precision", "recall", "f1"]}


def main():
    parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = Path("saved_models")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find latest UNet3D model
    model_files = sorted(model_dir.glob("best_unet3d_*.pt"), reverse=True)
    if not model_files:
        raise FileNotFoundError("No UNet3D model files found in saved_models/")

    model_path = model_files[0]
    print(f"Using model: {model_path}")

    # Load model
    model = load_model_checkpoint(model_path, device)

    # Get validation pairs (reuse same split logic)
    from src.data_manager import CSV_PATH, TRAIN_DIR, build_train_pairs
    import pandas as pd
    import random

    df = pd.read_csv(CSV_PATH, sep=";")
    split_col = next(c for c in df.columns if c.strip().lower() == "train/test/validation")
    train_df = df[df[split_col].astype(str).str.strip().eq("Train")]
    all_pairs = build_train_pairs(TRAIN_DIR, train_df)
    random.seed(42)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.8)
    val_pairs = all_pairs[split_idx:]

    # Evaluate
    results = evaluate_validation_3d(model, val_pairs, device)

    print(f"\n{'='*60}")
    print("3D UNet Validation Results")
    print(f"{'='*60}")
    print(f"  IoU:       {results['iou']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")

    # Save results
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"inference_3d_{ts}.json"

    with output_path.open("w") as f:
        json.dump({
            "model_path": str(model_path),
            "timestamp": ts,
            "device": str(device),
            "metrics": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
