import os
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_mvp.dataset import BraTS2DAxialDataset


# Constants
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(REPO_ROOT, "training_data1_v2")
CSV_PATH = os.path.join(REPO_ROOT, "validated_filtered.csv")


def build_train_pairs(train_dir: str, df: pd.DataFrame) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        subject_id = row["BraTS Subject ID"].strip()
        subj_dir = os.path.join(train_dir, subject_id)
        # modality: prefer T2-FLAIR (-t2f) as present in this repo; fall back to "-flair" if exists
        candidates = [
            os.path.join(subj_dir, f"{subject_id}-t2f.nii.gz"),
            os.path.join(subj_dir, f"{subject_id}-flair.nii.gz"),
        ]
        img_path = next((p for p in candidates if os.path.exists(p)), None)
        seg_path = os.path.join(subj_dir, f"{subject_id}-seg.nii.gz")
        if img_path and os.path.exists(seg_path):
            pairs.append((img_path, seg_path))
        else:
            # Skip missing pairs silently, or log if desired
            # print(f"Skipping missing files for {subject_id}")
            pass
    return pairs


def main():
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
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # Verification step: fetch one batch
    batch = next(iter(loader))
    images, labels = batch
    print("Batch images shape:", images.shape)  # [B,1,H,W]
    print("Batch labels shape:", labels.shape)  # [B,1,H,W]
    print("Images dtype:", images.dtype, "Labels dtype:", labels.dtype)


if __name__ == "__main__":
    main()
