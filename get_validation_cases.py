"""
Quick script to identify cases in the validation set.
Uses the same deterministic split as training (seed=42, val_split=0.2)
"""
import os
import random
import pandas as pd
from pathlib import Path

# Constants (matching dataManager.py)
REPO_ROOT = Path(__file__).parent
TRAIN_DIR = Path("/home/sg624ew/glioma_data/filtered_dataset")
CSV_PATH = REPO_ROOT / "validated_filtered.csv"

def get_validation_case_ids(val_split=0.2):
    """Get list of case IDs in the validation set."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    
    # Load CSV
    df = pd.read_csv(CSV_PATH, sep=";")
    
    # Find the split column
    split_col = None
    for c in df.columns:
        if c.strip().lower() == "train/test/validation":
            split_col = c
            break
    
    if split_col is None:
        raise KeyError("Could not find 'Train/Test/Validation' column in CSV")
    
    # Get training cases
    train_df = df[df[split_col].astype(str).str.strip().eq("Train")]
    
    # Build pairs (to match dataManager.py logic)
    all_train_ids = []
    for _, row in train_df.iterrows():
        subject_id = str(row["BraTS Subject ID"]).strip()
        if not subject_id:
            continue
        subj_dir = TRAIN_DIR / subject_id
        
        # Check if files exist
        candidates = [
            subj_dir / f"{subject_id}-t2f.nii.gz",
            subj_dir / f"{subject_id}-flair.nii.gz",
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        seg_path = subj_dir / f"{subject_id}-seg.nii.gz"
        
        if img_path and seg_path.exists():
            all_train_ids.append(subject_id)
    
    # Apply the same split logic as dataManager.py
    random.seed(42)  # Same seed as training
    random.shuffle(all_train_ids)
    
    split_idx = int(len(all_train_ids) * (1 - val_split))
    train_ids = all_train_ids[:split_idx]
    val_ids = all_train_ids[split_idx:]
    
    return val_ids, train_ids

if __name__ == "__main__":
    val_ids, train_ids = get_validation_case_ids()
    
    print(f"Total training cases: {len(train_ids)}")
    print(f"Total validation cases: {len(val_ids)}")
    print(f"\nFirst 10 validation case IDs:")
    print("-" * 50)
    for case_id in val_ids[:10]:
        print(f"  {case_id}")
    
    print(f"\n... and {len(val_ids) - 10} more validation cases")
    print(f"\nLast 5 validation case IDs:")
    print("-" * 50)
    for case_id in val_ids[-5:]:
        print(f"  {case_id}")
