"""
Single-case prediction utilities for trained segmentation models.
"""
import os
from pathlib import Path
from typing import Tuple, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from src.model import UNet


def load_and_prepare_volume(
    img_path: str,
    orientation: str = "axial"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a 3D NIfTI volume and prepare it for model inference.
    
    Args:
        img_path: Path to the image NIfTI file
        orientation: One of 'axial', 'coronal', 'sagittal'
    
    Returns:
        Tuple of (raw_volume, prepared_volume) both as numpy arrays
        raw_volume: Original volume in canonical orientation (H, W, D)
        prepared_volume: Reoriented volume with slices along axis 0 (num_slices, H, W)
    """
    axes = {"axial": 2, "coronal": 1, "sagittal": 0}
    if orientation.lower() not in axes:
        raise ValueError(f"Unsupported orientation '{orientation}'. Choose from axial, coronal, sagittal.")
    
    slice_axis = axes[orientation.lower()]
    
    # Load the volume
    img = nib.load(img_path)
    raw_volume = img.get_fdata(dtype=np.float32)
    
    # Reorient so slices are along axis 0
    prepared_volume = np.moveaxis(raw_volume, slice_axis, 0)
    
    return raw_volume, prepared_volume


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D slice using z-score normalization (matching training preprocessing).
    
    Args:
        slice_2d: 2D numpy array (H, W)
    
    Returns:
        Normalized 2D array
    """
    slice_2d = slice_2d.astype(np.float32)
    mean = slice_2d.mean()
    std = slice_2d.std()
    if std > 0:
        slice_2d = (slice_2d - mean) / std
    return slice_2d


def prepare_batch_for_model(image_slice: np.ndarray) -> torch.Tensor:
    """
    Prepare a single 2D slice for model input with proper padding.
    
    Args:
        image_slice: 2D numpy array (H, W)
    
    Returns:
        Torch tensor [1, 1, H_padded, W_padded]
    """
    # Normalize
    normalized = normalize_slice(image_slice)
    
    # Convert to tensor and add batch + channel dims
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Pad to be divisible by 16
    _, _, height, width = tensor.shape
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        pad = (0, pad_w, 0, pad_h)
        tensor = F.pad(tensor, pad, mode="reflect")
    
    return tensor


def unpad_prediction(prediction: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
    """
    Remove padding from model prediction to match original image shape.
    
    Args:
        prediction: Tensor [1, 1, H_padded, W_padded]
        original_shape: (H, W) of original image
    
    Returns:
        Numpy array (H, W) with values in [0, 1]
    """
    # Remove batch and channel dims
    pred = prediction.squeeze().cpu().numpy()
    
    # Crop to original size
    h, w = original_shape
    pred = pred[:h, :w]
    
    return pred


def predict_single_case(
    case_id: str,
    orientation: str,
    model_path: str,
    data_dir: str,
    device: Optional[torch.device] = None,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single case and return 3D prediction.
    
    Args:
        case_id: BraTS case ID (e.g., 'BraTS-GLI-00005-100')
        orientation: 'axial', 'coronal', or 'sagittal'
        model_path: Path to saved model checkpoint (.pt file)
        data_dir: Directory containing the case data
        device: torch device (defaults to CUDA if available)
        threshold: Probability threshold for binary prediction (default 0.5)
    
    Returns:
        Tuple of (prediction_volume, probability_volume)
        prediction_volume: Binary 3D array in canonical orientation (H, W, D)
        probability_volume: Probability 3D array in canonical orientation (H, W, D)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find the image file
    case_dir = Path(data_dir) / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    
    # Try to find T2-FLAIR image (matching training setup)
    candidates = [
        case_dir / f"{case_id}-t2f.nii.gz",
        case_dir / f"{case_id}-flair.nii.gz",
    ]
    img_path = next((str(p) for p in candidates if p.exists()), None)
    if img_path is None:
        raise FileNotFoundError(f"No T2-FLAIR image found for {case_id}")
    
    # Load and prepare volume
    raw_volume, prepared_volume = load_and_prepare_volume(img_path, orientation)
    num_slices, height, width = prepared_volume.shape
    
    # Load model
    model = UNet(in_channels=1, out_classes=1, up_sample_mode='conv_transpose').to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare output arrays
    prob_slices = []
    
    print(f"Running inference on {num_slices} slices for {orientation} orientation...")
    
    # Process each slice
    with torch.no_grad():
        for i in range(num_slices):
            slice_2d = prepared_volume[i]
            
            # Prepare input
            input_tensor = prepare_batch_for_model(slice_2d).to(device)
            
            # Run model
            logits = model(input_tensor)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            
            # Unpad and store
            prob_slice = unpad_prediction(probs, (height, width))
            prob_slices.append(prob_slice)
    
    # Stack slices back into 3D volume (num_slices, H, W)
    prob_volume_oriented = np.stack(prob_slices, axis=0)
    
    # Reorient back to canonical orientation (H, W, D)
    axes = {"axial": 2, "coronal": 1, "sagittal": 0}
    slice_axis = axes[orientation.lower()]
    prob_volume_canonical = np.moveaxis(prob_volume_oriented, 0, slice_axis)
    
    # Create binary prediction
    pred_volume_canonical = (prob_volume_canonical > threshold).astype(np.uint8)
    
    print(f"✓ Inference complete. Prediction shape: {pred_volume_canonical.shape}")
    
    return pred_volume_canonical, prob_volume_canonical


def ensemble_predictions(
    predictions: list,
    method: str = "average"
) -> np.ndarray:
    """
    Combine predictions from multiple orientations.
    
    Args:
        predictions: List of 3D probability volumes
        method: 'average' or 'voting'
    
    Returns:
        Combined binary prediction volume
    """
    if method == "average":
        # Average probabilities then threshold
        avg_probs = np.mean(predictions, axis=0)
        return (avg_probs > 0.5).astype(np.uint8)
    elif method == "voting":
        # Majority voting on binary predictions
        # Each prediction should already be binary
        votes = np.sum(predictions, axis=0)
        return (votes > len(predictions) / 2).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Dice coefficient between prediction and ground truth.
    
    Args:
        prediction: Binary prediction volume
        ground_truth: Binary ground truth volume
    
    Returns:
        Dice score in [0, 1]
    """
    # Flatten arrays
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    
    # Calculate intersection and union
    intersection = np.sum(pred_flat * gt_flat)
    total = np.sum(pred_flat) + np.sum(gt_flat)
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2 * intersection / total
    return dice
