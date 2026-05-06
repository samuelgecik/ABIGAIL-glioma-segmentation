"""
Generate persistent train/val/test splits for the BraTS-GLI dataset.

Reads validated_filtered.csv, performs a stratified split by glioma type
(70% train, 15% val, 15% test), and writes dataset_splits.csv with an
explicit Split column. The random seed is fixed for reproducibility.

Usage:
    python generate_splits.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
INPUT_CSV = "validated_filtered.csv"
OUTPUT_CSV = "dataset_splits.csv"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    df = pd.read_csv(INPUT_CSV, sep=";")

    # Normalize column names (some have trailing spaces)
    df.columns = [c.strip() for c in df.columns]

    glioma_col = "Glioma type"
    subject_col = "BraTS Subject ID"

    print(f"Total subjects: {len(df)}")
    print(f"Glioma distribution:\n{df[glioma_col].value_counts().to_string()}\n")

    # First split: train (70%) vs. temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=df[glioma_col],
    )

    # Second split: val (50% of temp = 15%) vs. test (50% of temp = 15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=temp_df[glioma_col],
    )

    # Assign split labels
    df["Split"] = None
    df.loc[train_df.index, "Split"] = "Train"
    df.loc[val_df.index, "Split"] = "Val"
    df.loc[test_df.index, "Split"] = "Test"

    assert df["Split"].notna().all(), "Some subjects were not assigned a split"

    # Replace the old Train/Test/Validation column with the new Split
    if "Train/Test/Validation" in df.columns:
        df = df.drop(columns=["Train/Test/Validation"])

    # Sort by subject ID for stable ordering
    df = df.sort_values(subject_col).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, sep=";", index=False)
    print(f"Wrote {OUTPUT_CSV}")

    # Print summary
    print(f"\n{'Split':<8} {'Count':>6} {'%':>7}")
    print("-" * 23)
    for split in ["Train", "Val", "Test"]:
        subset = df[df["Split"] == split]
        pct = 100 * len(subset) / len(df)
        print(f"{split:<8} {len(subset):>6} {pct:>6.1f}%")

    # Verify stratification
    print(f"\nStratification check (glioma type % per split):")
    print("-" * 55)
    header = f"{'Type':<22}"
    for split in ["Train", "Val", "Test"]:
        header += f" {split:>8}"
    print(header)
    print("-" * 55)
    for gtype in sorted(df[glioma_col].unique()):
        row = f"{gtype:<22}"
        for split in ["Train", "Val", "Test"]:
            subset = df[df["Split"] == split]
            count = (subset[glioma_col] == gtype).sum()
            pct = 100 * count / len(subset)
            row += f" {pct:>7.1f}%"
        print(row)


if __name__ == "__main__":
    main()
