"""
Volumetric 3-axis ensemble evaluation.

Runs the per-orientation 2D models (axial / coronal / sagittal) over every
volume in a split, fuses their probabilities in canonical 3D space, and reports
*volumetric* (per-patient 3D) metrics — the number that reflects real
deployment, as opposed to the per-slice-per-axis metrics from ``inference.py``.

Usage:
    python -m src.ensemble_inference --model-arch segformer --split Val
    python -m src.ensemble_inference --model-arch segformer --split Test --threshold 0.5
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from src.data_manager import CSV_PATH, DATA_DIR, _load_splits
from src.model import build_model, VALID_ARCHITECTURES
from src.predict import (
    load_and_prepare_volume,
    prepare_batch_for_model,
    unpad_prediction,
)

ORIENTATIONS = ("axial", "coronal", "sagittal")
# Axis (in canonical H, W, D space) along which each orientation slices.
SLICE_AXES = {"axial": 2, "coronal": 1, "sagittal": 0}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volumetric 3-axis ensemble evaluation of trained segmentation models."
    )
    parser.add_argument(
        "--model-arch", type=str, default="unet", choices=VALID_ARCHITECTURES,
        help="Model architecture to evaluate (default: unet)",
    )
    parser.add_argument(
        "--split", type=str, default="Val", choices=("Train", "Val", "Test"),
        help="Dataset split to evaluate on (default: Val)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for the fused binary mask (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Slices per forward pass within a volume (default: 32)",
    )
    parser.add_argument(
        "--sweep", type=str, default=None,
        help="Comma-separated thresholds to evaluate the ensemble at in a single "
             "pass, e.g. '0.3,0.4,0.5,0.6,0.7'. Reports per-case mean metrics at "
             "each and the best by Dice. Overrides --threshold for the report.",
    )
    parser.add_argument(
        "--timestamp", type=str, default=None,
        help="Evaluate a specific model-set timestamp (e.g. '20260610-220000') "
             "instead of the most recent complete set.",
    )
    return parser.parse_args()


def find_model_set(model_dir: Path, model_arch: str, timestamp: str = None) -> Tuple[str, Dict[str, Path]]:
    """Find a complete {axial, coronal, sagittal} checkpoint set.

    If ``timestamp`` is given, return that exact set; otherwise the most recent
    complete set. Mirrors the file-naming logic in ``inference.py``:
        UNet:   best_{orientation}_{timestamp}.pt        (no arch prefix)
        Others: best_{arch}_{orientation}_{timestamp}.pt
    """
    if model_arch == "unet":
        glob_pattern = "best_*.pt"
        arch_prefixes = tuple(f"best_{a}_" for a in VALID_ARCHITECTURES if a != "unet")
        prefix_parts = 1
    else:
        glob_pattern = f"best_{model_arch}_*.pt"
        arch_prefixes = None
        prefix_parts = 2

    files = list(model_dir.glob(glob_pattern))
    if arch_prefixes:
        files = [f for f in files if not f.name.startswith(arch_prefixes)]
    if not files:
        raise FileNotFoundError(f"No {model_arch} model files found in {model_dir}")

    groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= prefix_parts + 2:
            orientation = parts[prefix_parts]
            timestamp = "_".join(parts[prefix_parts + 1:])
            groups[timestamp].append((orientation, f))

    if timestamp is not None:
        if timestamp not in groups:
            raise FileNotFoundError(
                f"No {model_arch} models found for timestamp '{timestamp}' in {model_dir}"
            )
        by_ori = {ori: path for ori, path in groups[timestamp]}
        missing = set(ORIENTATIONS) - set(by_ori)
        if missing:
            raise FileNotFoundError(
                f"Incomplete {model_arch} set for timestamp '{timestamp}': missing {sorted(missing)}"
            )
        return timestamp, {ori: by_ori[ori] for ori in ORIENTATIONS}

    for ts, models in sorted(groups.items(), reverse=True):
        by_ori = {ori: path for ori, path in models}
        if set(ORIENTATIONS).issubset(by_ori):
            return ts, {ori: by_ori[ori] for ori in ORIENTATIONS}

    raise FileNotFoundError(
        f"Could not find a complete set of {model_arch} models (axial, coronal, sagittal)"
    )


def load_orientation_models(
    model_set: Dict[str, Path], model_arch: str, device: torch.device
) -> Dict[str, torch.nn.Module]:
    """Load all three orientation checkpoints once, up front."""
    models = {}
    for ori in ORIENTATIONS:
        ckpt = torch.load(model_set[ori], map_location=device)
        arch = ckpt.get("model_arch", model_arch)
        model = build_model(arch, in_channels=1, out_classes=1).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models[ori] = model
        print(f"  {ori:9s}: {model_set[ori].name}  (epoch {ckpt.get('epoch', '?')}, "
              f"val_iou {ckpt.get('val_iou', float('nan')):.4f})")
    return models


@torch.no_grad()
def predict_probability_volume(
    model: torch.nn.Module,
    prepared_volume: np.ndarray,  # (num_slices, H, W) along the orientation axis
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run a model over all slices of one (already-reoriented) volume.

    Returns a probability volume of the same (num_slices, H, W) shape.
    """
    num_slices, height, width = prepared_volume.shape
    # All slices in this volume share a shape, so we pad once and batch.
    tensors = [prepare_batch_for_model(prepared_volume[i]) for i in range(num_slices)]
    batch_full = torch.cat(tensors, dim=0)  # (num_slices, 1, Hp, Wp)

    prob_slices = []
    for start in range(0, num_slices, batch_size):
        chunk = batch_full[start:start + batch_size].to(device)
        logits = model(chunk)
        probs = torch.sigmoid(logits).cpu()
        for j in range(probs.shape[0]):
            prob_slices.append(unpad_prediction(probs[j:j + 1], (height, width)))
    return np.stack(prob_slices, axis=0)


def to_canonical(prob_oriented: np.ndarray, orientation: str) -> np.ndarray:
    """Move the slice axis back to its canonical position (H, W, D)."""
    return np.moveaxis(prob_oriented, 0, SLICE_AXES[orientation])


def volumetric_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Binary volumetric metrics for one 3D case."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())

    dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0
    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def _mean_metrics(per_case: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_case:
        return {k: float("nan") for k in ("dice", "iou", "precision", "recall")}
    keys = per_case[0].keys()
    return {k: float(np.mean([c[k] for c in per_case])) for k in keys}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model architecture: {args.model_arch} | Split: {args.split} | Threshold: {args.threshold}")

    # Resolve checkpoint set
    model_dir = Path("saved_models")
    timestamp, model_set = find_model_set(model_dir, args.model_arch, args.timestamp)
    print(f"\nUsing model set with timestamp: {timestamp}")
    models = load_orientation_models(model_set, args.model_arch, device)

    # Gather subjects for the requested split
    df = _load_splits()
    subject_ids = [str(s).strip() for s in df[df["Split"] == args.split]["BraTS Subject ID"].tolist()]
    if not subject_ids:
        raise RuntimeError(f"No subjects found for split '{args.split}' in {CSV_PATH}")

    # Thresholds to evaluate the ensemble at. In sweep mode we test several in
    # one pass; otherwise just the single --threshold.
    if args.sweep:
        sweep_thresholds = [float(t) for t in args.sweep.split(",") if t.strip()]
    else:
        sweep_thresholds = [args.threshold]

    # Accumulators: per-case metrics for the single axes + the ensemble at each threshold
    per_axis_cases: Dict[str, List[Dict[str, float]]] = {o: [] for o in ORIENTATIONS}
    ensemble_cases_by_t: Dict[float, List[Dict[str, float]]] = {t: [] for t in sweep_thresholds}
    skipped: List[str] = []

    print(f"\nEvaluating {len(subject_ids)} subjects from split '{args.split}'...")
    for case_id in tqdm(subject_ids, desc="Ensemble eval", unit="case"):
        case_dir = Path(DATA_DIR) / case_id
        img_candidates = [case_dir / f"{case_id}-t2f.nii.gz", case_dir / f"{case_id}-flair.nii.gz"]
        img_path = next((p for p in img_candidates if p.exists()), None)
        seg_path = case_dir / f"{case_id}-seg.nii.gz"
        if img_path is None or not seg_path.exists():
            skipped.append(case_id)
            continue

        gt = (nib.load(str(seg_path)).get_fdata() != 0).astype(np.uint8)  # binary whole-tumor

        prob_sum = None
        for ori in ORIENTATIONS:
            _, prepared = load_and_prepare_volume(str(img_path), ori)
            prob_oriented = predict_probability_volume(models[ori], prepared, device, args.batch_size)
            prob_canon = to_canonical(prob_oriented, ori)
            if prob_canon.shape != gt.shape:
                raise ValueError(f"{case_id}: prob shape {prob_canon.shape} != gt {gt.shape}")

            # Single-axis volumetric metrics at the primary threshold (for reference)
            axis_pred = (prob_canon > args.threshold).astype(np.uint8)
            per_axis_cases[ori].append(volumetric_metrics(axis_pred, gt))

            prob_sum = prob_canon if prob_sum is None else prob_sum + prob_canon

        # Ensemble = mean probability across the 3 axes; evaluate at every threshold
        # (computed once; thresholding is cheap, so the sweep adds negligible cost).
        ensemble_prob = prob_sum / len(ORIENTATIONS)
        for t in sweep_thresholds:
            pred = (ensemble_prob > t).astype(np.uint8)
            ensemble_cases_by_t[t].append(volumetric_metrics(pred, gt))

    # Aggregate
    per_axis_avg = {o: _mean_metrics(per_axis_cases[o]) for o in ORIENTATIONS}
    ensemble_avg_by_t = {t: _mean_metrics(ensemble_cases_by_t[t]) for t in sweep_thresholds}
    best_t = max(sweep_thresholds, key=lambda t: ensemble_avg_by_t[t]["dice"])
    n_cases = len(ensemble_cases_by_t[sweep_thresholds[0]])

    def _fmt(m):
        return f"dice={m['dice']:.4f}  iou={m['iou']:.4f}  prec={m['precision']:.4f}  recall={m['recall']:.4f}"

    print(f"\n{'='*70}")
    print(f"VOLUMETRIC RESULTS  ({n_cases} cases, split={args.split})")
    print(f"{'='*70}")
    for o in ORIENTATIONS:
        print(f"  {o:9s} (single-axis @ {args.threshold:.2f}): {_fmt(per_axis_avg[o])}")
    print(f"  {'-'*64}")
    if args.sweep:
        print("  ENSEMBLE (3-axis) threshold sweep:")
        for t in sweep_thresholds:
            marker = "  <-- best Dice" if t == best_t else ""
            print(f"    t={t:.2f}: {_fmt(ensemble_avg_by_t[t])}{marker}")
    else:
        print(f"  {'ENSEMBLE':9s} (3-axis @ {args.threshold:.2f}): {_fmt(ensemble_avg_by_t[best_t])}")
    if skipped:
        print(f"\n  NOTE: skipped {len(skipped)} subjects missing image/seg files: {skipped[:5]}"
              + (" ..." if len(skipped) > 5 else ""))

    # Persist
    out_dir = Path("inference_results")
    out_dir.mkdir(exist_ok=True)
    now = time.strftime("%Y%m%d-%H%M%S")
    tag = "ensemblesweep" if args.sweep else "ensemble"
    out_path = out_dir / f"{tag}_{args.model_arch}_{args.split}_{timestamp}_{now}.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "model_arch": args.model_arch,
                "model_timestamp": timestamp,
                "split": args.split,
                "primary_threshold": args.threshold,
                "swept": bool(args.sweep),
                "best_threshold_by_dice": best_t,
                "num_cases": n_cases,
                "num_skipped": len(skipped),
                "skipped_ids": skipped,
                "metric_type": "volumetric_per_case_mean",
                "per_axis": per_axis_avg,
                "ensemble_by_threshold": {f"{t:.2f}": ensemble_avg_by_t[t] for t in sweep_thresholds},
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
