import os
import random
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import binarize_mask

# Shared volume cache with the 2D dataset
from src.dataset import _volume_cache


class RandomAugment3D:
    """Random augmentations for 3D medical image volumes."""

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # image: [1, D, H, W], label: [1, D, H, W]

        # Random axis flips (spatial dims 1, 2, 3)
        for axis in (1, 2, 3):
            if random.random() < 0.5:
                image = torch.flip(image, [axis])
                label = torch.flip(label, [axis])

        # Random 90-degree rotation — only in planes where both dims are equal
        # For BraTS 182x218x182: D-W plane (dims 1,3) is 182x182 (square)
        square_planes = []
        if image.shape[1] == image.shape[2]:  # D == H
            square_planes.append((1, 2))
        if image.shape[1] == image.shape[3]:  # D == W
            square_planes.append((1, 3))
        if image.shape[2] == image.shape[3]:  # H == W
            square_planes.append((2, 3))
        if square_planes and random.random() < 0.25:
            plane = random.choice(square_planes)
            k = random.choice([1, 2, 3])
            image = torch.rot90(image, k, list(plane))
            label = torch.rot90(label, k, list(plane))

        # Random intensity scaling (image only)
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            image = image * scale

        # Random intensity shift (image only)
        if random.random() < 0.3:
            shift = random.uniform(-0.1, 0.1)
            image = image + shift

        # Random Gaussian noise (image only)
        if random.random() < 0.2:
            std = random.uniform(0.0, 0.1)
            image = image + torch.randn_like(image) * std

        return image, label


class BraTS3DDataset(Dataset):
    """
    Return full 3D NIfTI volumes as tensors.

    Each item returns (image_tensor, label_tensor) shaped [1, D, H, W].
    Volume-level z-score normalization over brain (nonzero) voxels.
    """

    def __init__(
        self,
        file_paths: List[Tuple[str, str]],
        transforms: Optional[RandomAugment3D] = None,
        preload: bool = True,
    ):
        super().__init__()
        self.file_paths = file_paths
        self.transforms = transforms
        self.preload = preload
        self.volumes: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []

        print(f"Loading {len(file_paths)} 3D volumes...")
        for vol_idx, (img_path, lbl_path) in enumerate(file_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image file: {img_path}")
            if not os.path.exists(lbl_path):
                raise FileNotFoundError(f"Missing label file: {lbl_path}")

            if preload:
                img_raw = self._load_cached(img_path, dtype=np.float32)
                lbl_raw = self._load_cached(lbl_path, dtype=np.int8)
                if img_raw.shape != lbl_raw.shape:
                    raise ValueError(
                        f"Shape mismatch: {img_path} {img_raw.shape} vs {lbl_path} {lbl_raw.shape}"
                    )
                self.volumes.append((img_raw, lbl_raw))
            else:
                self.volumes.append(None)

            if preload and (vol_idx + 1) % 100 == 0:
                print(f"  Loaded {vol_idx + 1}/{len(file_paths)} volumes...")

        if preload:
            print(f"  All {len(file_paths)} volumes loaded.")

    @staticmethod
    def _load_cached(path: str, dtype) -> np.ndarray:
        if path in _volume_cache:
            arr = _volume_cache[path]
        else:
            arr = nib.load(path).get_fdata(dtype=np.float32)
            if dtype != np.float32:
                arr = arr.astype(dtype)
            _volume_cache[path] = arr
        return arr if arr.dtype == dtype else arr.astype(dtype)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        if self.preload:
            img_raw, lbl_raw = self.volumes[idx]
        else:
            img_path, lbl_path = self.file_paths[idx]
            img_raw = nib.load(img_path).get_fdata(dtype=np.float32)
            lbl_raw = nib.load(lbl_path).get_fdata(dtype=np.float32).astype(np.int8)

        # Volume-level z-score normalization over brain voxels
        image = img_raw.astype(np.float32)
        brain_mask = image > 0
        if brain_mask.any():
            brain_voxels = image[brain_mask]
            mean = brain_voxels.mean()
            std = brain_voxels.std()
            if std > 0:
                image = (image - mean) / std
                image[~brain_mask] = 0.0  # keep background at 0

        # Binarize label
        label = binarize_mask(lbl_raw.astype(np.int16))

        # To tensors with channel dim: [1, D, H, W]
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(torch.float32)
        label_tensor = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)

        if self.transforms is not None:
            image_tensor, label_tensor = self.transforms(image_tensor, label_tensor)

        return image_tensor, label_tensor
