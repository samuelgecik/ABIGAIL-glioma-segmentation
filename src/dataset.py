import os
from typing import List, Optional, Tuple, Dict

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import binarize_mask


class BraTS2DDataset(Dataset):
    """
    Turn a list of (image_path, label_path) 3D NIfTI volumes into per-slice 2D samples.

    Each item returns a tuple (image_tensor, label_tensor) shaped [1, H, W].
    Preloads all volumes into memory for maximum speed.
    """

    def __init__(self, file_paths: List[Tuple[str, str]], transforms: Optional[object] = None, orientation: str = "axial", preload: bool = True):
        super().__init__()
        self.file_paths = file_paths
        self.transforms = transforms
        self.slice_map: List[Tuple[int, int]] = []  # (volume_idx, slice_idx)
        axes = {"axial": 2, "coronal": 1, "sagittal": 0}
        orientation_lc = orientation.lower()
        if orientation_lc not in axes:
            raise ValueError(f"Unsupported orientation '{orientation}'. Choose from axial, coronal, sagittal.")
        self.slice_axis = axes[orientation_lc]
        
        # Preload all volumes into memory (shared across workers)
        self.volumes: List[Tuple[np.ndarray, np.ndarray]] = []
        self.preload = preload

        print(f"Loading {len(file_paths)} volumes into memory...")
        # Pre-scan volumes to build a global slice index map
        for vol_idx, (img_path, lbl_path) in enumerate(file_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image file: {img_path}")
            if not os.path.exists(lbl_path):
                raise FileNotFoundError(f"Missing label file: {lbl_path}")

            if preload:
                # Load volume immediately
                img_vol = np.moveaxis(nib.load(img_path).get_fdata(dtype=np.float32), self.slice_axis, 0)
                lbl_vol = np.moveaxis(nib.load(lbl_path).get_fdata(dtype=np.float32), self.slice_axis, 0).astype(np.int16)
                
                if img_vol.shape != lbl_vol.shape:
                    raise ValueError(
                        f"Image/Label shape mismatch: {img_path} {img_vol.shape} vs {lbl_path} {lbl_vol.shape}"
                    )
                if img_vol.ndim != 3:
                    raise ValueError(f"Expected 3D volume, got {img_vol.shape}")
                
                depth = img_vol.shape[0]
                self.volumes.append((img_vol, lbl_vol))
            else:
                # Use header shape to avoid loading full data into memory
                img = nib.load(img_path)
                img_shape = img.shape
                if len(img_shape) != 3:
                    raise ValueError(f"Expected 3D image volume at {img_path}, got shape {img_shape}")
                depth = img_shape[self.slice_axis]
                self.volumes.append(None)  # Placeholder

            for z in range(depth):
                self.slice_map.append((vol_idx, z))
            
            if preload and (vol_idx + 1) % 100 == 0:
                print(f"  Loaded {vol_idx + 1}/{len(file_paths)} volumes...")
        
        if preload:
            print(f"✓ All {len(file_paths)} volumes loaded into memory!")

    def __len__(self) -> int:
        return len(self.slice_map)

    def __getitem__(self, idx: int):
        vol_idx, z = self.slice_map[idx]
        
        if self.preload:
            img_vol, lbl_vol = self.volumes[vol_idx]
        else:
            # Fallback: load on demand
            img_path, lbl_path = self.file_paths[vol_idx]
            img_vol = np.moveaxis(nib.load(img_path).get_fdata(dtype=np.float32), self.slice_axis, 0)
            lbl_vol = np.moveaxis(nib.load(lbl_path).get_fdata(dtype=np.float32), self.slice_axis, 0).astype(np.int16)

        image_slice = img_vol[z].astype(np.float32)
        label_slice = lbl_vol[z].astype(np.int16)

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
