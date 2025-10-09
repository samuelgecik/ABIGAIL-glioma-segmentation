import numpy as np


def binarize_mask(mask_slice: np.ndarray) -> np.ndarray:
    """
    Convert a 2D segmentation slice to binary: any non-zero becomes 1.

    Args:
        mask_slice: 2D numpy array (H, W) with integer labels.

    Returns:
        2D numpy array (H, W) of dtype uint8 with values {0,1}.
    """
    if mask_slice is None:
        raise ValueError("mask_slice cannot be None")
    if mask_slice.ndim != 2:
        raise ValueError(f"mask_slice must be 2D, got shape {mask_slice.shape}")
    return (mask_slice != 0).astype(np.uint8)
