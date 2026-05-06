"""
Quick script to list cases in a given split (Train, Val, or Test).
Reads from the persistent dataset_splits.csv.
"""
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent
CSV_PATH = REPO_ROOT / "dataset_splits.csv"


def get_case_ids(split: str = "Val") -> list[str]:
    """Get list of case IDs for the given split."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Split CSV not found at {CSV_PATH}. "
            "Run `python generate_splits.py` to create it."
        )
    df = pd.read_csv(CSV_PATH, sep=";")
    subset = df[df["Split"] == split]
    return subset["BraTS Subject ID"].tolist()


if __name__ == "__main__":
    for split in ["Train", "Val", "Test"]:
        ids = get_case_ids(split)
        print(f"{split}: {len(ids)} cases")

    print("\nValidation case IDs:")
    print("-" * 50)
    val_ids = get_case_ids("Val")
    for case_id in val_ids[:10]:
        print(f"  {case_id}")
    if len(val_ids) > 10:
        print(f"  ... and {len(val_ids) - 10} more")
