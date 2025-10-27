import os
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_mvp.dataset import BraTS2DAxialDataset


# Constants
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(REPO_ROOT, "training_data1_v2")
VALIDATION_DIR = os.path.join(REPO_ROOT, "validation_data")
CSV_PATH = os.path.join(REPO_ROOT, "validated_filtered.csv")


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


def build_train_pairs(train_dir: str, df: pd.DataFrame) -> List[Tuple[str, str]]:
    subject_ids = (row["BraTS Subject ID"] for _, row in df.iterrows())
    return _build_pairs(subject_ids, train_dir)


def get_training_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    # The CSV is semicolon-delimited based on repo file
    df = pd.read_csv(CSV_PATH, sep=";")

    # Normalize column name (there is a trailing space in 'Train/Test/Validation ')
    split_col = None
    for c in df.columns:
        if c.strip().lower() == "train/test/validation":
            split_col = c
            break
    if split_col is None:
        raise KeyError("Could not find 'Train/Test/Validation' column in CSV")

    train_df = df[df[split_col].astype(str).str.strip().eq("Train")]

    train_pairs = build_train_pairs(TRAIN_DIR, train_df)
    if not train_pairs:
        raise RuntimeError("No training pairs found. Check paths and filenames.")

    dataset = BraTS2DAxialDataset(train_pairs)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # Verification step: fetch one batch
    batch = next(iter(loader))
    images, labels = batch
    print("Batch images shape:", images.shape)  # [B,1,H,W]
    print("Batch labels shape:", labels.shape)  # [B,1,H,W]
    print("Images dtype:", images.dtype, "Labels dtype:", labels.dtype)
    return loader


def get_validation_data():
    if not os.path.isdir(VALIDATION_DIR):
        raise FileNotFoundError(f"Validation directory not found at {VALIDATION_DIR}")

    subject_ids = [d for d in os.listdir(VALIDATION_DIR) if os.path.isdir(os.path.join(VALIDATION_DIR, d))]
    validation_pairs = _build_pairs(subject_ids, VALIDATION_DIR)
    if not validation_pairs:
        raise RuntimeError("No validation pairs found. Check validation data paths and filenames.")

    dataset = BraTS2DAxialDataset(validation_pairs)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    return loader


if __name__ == "__main__":
    get_training_data()
