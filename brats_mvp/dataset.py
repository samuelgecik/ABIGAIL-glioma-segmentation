import os
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from brats_mvp.utils import binarize_mask


class BraTS2DAxialDataset(Dataset):
    """
    Turn a list of (image_path, label_path) 3D NIfTI volumes into per-slice 2D samples.

    Each item returns a tuple (image_tensor, label_tensor) shaped [1, H, W].
    """

    def __init__(self, file_paths: List[Tuple[str, str]], transforms: Optional[object] = None):
        super().__init__()
        self.file_paths = file_paths
        self.transforms = transforms
        self.slice_map: List[Tuple[str, str, int]] = []

        # Pre-scan volumes to build a global slice index map
        for img_path, lbl_path in self.file_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image file: {img_path}")
            if not os.path.exists(lbl_path):
                raise FileNotFoundError(f"Missing label file: {lbl_path}")

            # Use header shape to avoid loading full data into memory here
            img = nib.load(img_path)
            img_shape = img.shape
            if len(img_shape) != 3:
                raise ValueError(f"Expected 3D image volume at {img_path}, got shape {img_shape}")

            depth = img_shape[2]  # axial slices along last dim
            for z in range(depth):
                self.slice_map.append((img_path, lbl_path, z))

    def __len__(self) -> int:
        return len(self.slice_map)

    def __getitem__(self, idx: int):
        img_path, lbl_path, z = self.slice_map[idx]

        img_vol = nib.load(img_path).get_fdata()
        lbl_vol = nib.load(lbl_path).get_fdata()

        # Safety checks
        if img_vol.shape != lbl_vol.shape:
            raise ValueError(
                f"Image/Label shape mismatch: {img_path} {img_vol.shape} vs {lbl_path} {lbl_vol.shape}"
            )
        if img_vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {img_vol.shape}")

        image_slice = img_vol[:, :, z].astype(np.float32)
        label_slice = lbl_vol[:, :, z].astype(np.int16)

        # Normalize image slice (simple z-score within-slice if std>0)
        mean = image_slice.mean()
        std = image_slice.std()
        if std > 0:
            image_slice = (image_slice - mean) / std

        # Binarize label
        label_slice = binarize_mask(label_slice)

        # To torch tensors with channel dim
        image_tensor = torch.from_numpy(image_slice).unsqueeze(0).to(torch.float32)  # [1,H,W]
        label_tensor = torch.from_numpy(label_slice).unsqueeze(0).to(torch.float32)  # [1,H,W]

        if self.transforms is not None:
            # Transforms may expect (image, mask) numpy arrays or tensors; we assume tensor API
            image_tensor, label_tensor = self.transforms(image_tensor, label_tensor)

        return image_tensor, label_tensor
