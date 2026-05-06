import os
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import BraTS2DDataset
from src.dataset3d import BraTS3DDataset, RandomAugment3D


# Constants
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.abspath("/home/sg624ew/glioma_data/filtered_dataset")  # Local SSD copy for faster I/O
CSV_PATH = os.path.join(REPO_ROOT, "dataset_splits.csv")


def _build_pairs(subject_ids, base_dir: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for raw_id in subject_ids:
        subject_id = str(raw_id).strip()
        if not subject_id:
            continue
        subj_dir = os.path.join(base_dir, subject_id)
        # modality: prefer T2-FLAIR (-t2f) as present in this repo; fall back to "-flair" if exists
        candidates = [
            os.path.join(subj_dir, f"{subject_id}-t2f.nii.gz"),
            os.path.join(subj_dir, f"{subject_id}-flair.nii.gz"),
        ]
        img_path = next((p for p in candidates if os.path.exists(p)), None)
        seg_path = os.path.join(subj_dir, f"{subject_id}-seg.nii.gz")
        if img_path and os.path.exists(seg_path):
            pairs.append((img_path, seg_path))
    return pairs


def _load_splits() -> pd.DataFrame:
    """Load the persistent split CSV and return the DataFrame."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Split CSV not found at {CSV_PATH}. "
            "Run `python generate_splits.py` to create it."
        )
    df = pd.read_csv(CSV_PATH, sep=";")
    if "Split" not in df.columns:
        raise KeyError("CSV is missing the 'Split' column. Regenerate with generate_splits.py.")
    return df


def _pairs_for_split(df: pd.DataFrame, split: str) -> List[Tuple[str, str]]:
    """Return (image_path, seg_path) pairs for a given split name."""
    subset = df[df["Split"] == split]
    subject_ids = subset["BraTS Subject ID"].tolist()
    pairs = _build_pairs(subject_ids, DATA_DIR)
    if not pairs:
        raise RuntimeError(
            f"No pairs found for split '{split}'. "
            f"Check that files exist in {DATA_DIR}."
        )
    return pairs


def get_training_data(orientation="axial", preload=True):
    """Load 2D training and validation data from persistent splits.

    Returns (train_loader, val_loader).
    """
    df = _load_splits()

    train_pairs = _pairs_for_split(df, "Train")
    val_pairs = _pairs_for_split(df, "Val")

    train_dataset = BraTS2DDataset(train_pairs, orientation=orientation, preload=preload)
    val_dataset = BraTS2DDataset(val_pairs, orientation=orientation, preload=preload)

    # Use num_workers=0 (main process only) since data is preloaded in RAM
    # Larger batch size to maximize GPU utilization with 24GB VRAM
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Verification step: fetch one batch
    batch = next(iter(train_loader))
    images, labels = batch
    print("Batch images shape:", images.shape)  # [B,1,H,W]
    print("Batch labels shape:", labels.shape)  # [B,1,H,W]
    print("Images dtype:", images.dtype, "Labels dtype:", labels.dtype)
    print(f"Train volumes: {len(train_pairs)}, Validation volumes: {len(val_pairs)}")
    print(f"Train slices: {len(train_dataset)}, Validation slices: {len(val_dataset)}")
    return train_loader, val_loader


def get_training_data_3d(preload=True, augment=True):
    """Load 3D volumetric training and validation data from persistent splits.

    Returns (train_loader, val_loader).
    """
    df = _load_splits()

    train_pairs = _pairs_for_split(df, "Train")
    val_pairs = _pairs_for_split(df, "Val")

    transforms = RandomAugment3D() if augment else None
    train_dataset = BraTS3DDataset(train_pairs, transforms=transforms, preload=preload)
    val_dataset = BraTS3DDataset(val_pairs, transforms=None, preload=preload)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    batch = next(iter(train_loader))
    images, labels = batch
    print("Batch images shape:", images.shape)  # [B,1,D,H,W]
    print("Batch labels shape:", labels.shape)
    print("Images dtype:", images.dtype, "Labels dtype:", labels.dtype)
    print(f"Train volumes: {len(train_pairs)}, Validation volumes: {len(val_pairs)}")
    return train_loader, val_loader


def get_test_data(orientation="axial", preload=False):
    """Load 2D test data from persistent splits.

    Returns a single DataLoader for the held-out test set.
    """
    df = _load_splits()
    test_pairs = _pairs_for_split(df, "Test")

    test_dataset = BraTS2DDataset(test_pairs, orientation=orientation, preload=preload)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Test volumes: {len(test_pairs)}, Test slices: {len(test_dataset)}")
    return test_loader


def get_test_data_3d(preload=False):
    """Load 3D volumetric test data from persistent splits.

    Returns a single DataLoader for the held-out test set.
    """
    df = _load_splits()
    test_pairs = _pairs_for_split(df, "Test")

    test_dataset = BraTS3DDataset(test_pairs, transforms=None, preload=preload)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Test volumes: {len(test_pairs)}")
    return test_loader


if __name__ == "__main__":
    get_training_data()
