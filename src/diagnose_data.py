"""
Quick diagnostic script to analyze class distribution and data quality.
"""

import torch
import numpy as np
from src.dataManager import get_training_data
from tqdm import tqdm


def analyze_class_distribution(orientation="axial", num_batches=100):
    """Analyze class distribution in the dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Data Analysis for {orientation.upper()} orientation")
    print(f"{'='*70}\n")
    
    train_loader, val_loader = get_training_data(val_split=0.2, orientation=orientation)
    
    # Training set analysis
    print("Analyzing TRAINING set...")
    train_positive = 0
    train_negative = 0
    train_slices_with_tumor = 0
    train_total_slices = 0
    
    for i, (images, labels) in enumerate(tqdm(train_loader, total=min(num_batches, len(train_loader)))):
        if i >= num_batches:
            break
        labels = labels.to(device)
        train_positive += labels.sum().item()
        train_negative += (1 - labels).sum().item()
        
        # Count slices with any tumor
        batch_size = labels.shape[0]
        for j in range(batch_size):
            train_total_slices += 1
            if labels[j].sum() > 0:
                train_slices_with_tumor += 1
    
    # Validation set analysis
    print("\nAnalyzing VALIDATION set...")
    val_positive = 0
    val_negative = 0
    val_slices_with_tumor = 0
    val_total_slices = 0
    
    for i, (images, labels) in enumerate(tqdm(val_loader, total=min(num_batches, len(val_loader)))):
        if i >= num_batches:
            break
        labels = labels.to(device)
        val_positive += labels.sum().item()
        val_negative += (1 - labels).sum().item()
        
        batch_size = labels.shape[0]
        for j in range(batch_size):
            val_total_slices += 1
            if labels[j].sum() > 0:
                val_slices_with_tumor += 1
    
    # Calculate statistics
    train_total = train_positive + train_negative
    val_total = val_positive + val_negative
    
    print(f"\n{'='*70}")
    print("TRAINING SET STATISTICS")
    print(f"{'='*70}")
    print(f"Total pixels analyzed:        {train_total:>15,}")
    print(f"Tumor pixels (positive):      {train_positive:>15,} ({100*train_positive/train_total:>5.2f}%)")
    print(f"Background pixels (negative): {train_negative:>15,} ({100*train_negative/train_total:>5.2f}%)")
    print(f"Class imbalance ratio:        {train_negative/train_positive:>15.1f}:1")
    print(f"Recommended pos_weight:       {train_negative/train_positive:>15.1f}")
    print(f"\nSlices with tumor:            {train_slices_with_tumor:>15,} / {train_total_slices:,} ({100*train_slices_with_tumor/train_total_slices:.1f}%)")
    print(f"Empty slices (no tumor):      {train_total_slices - train_slices_with_tumor:>15,} ({100*(train_total_slices-train_slices_with_tumor)/train_total_slices:.1f}%)")
    
    print(f"\n{'='*70}")
    print("VALIDATION SET STATISTICS")
    print(f"{'='*70}")
    print(f"Total pixels analyzed:        {val_total:>15,}")
    print(f"Tumor pixels (positive):      {val_positive:>15,} ({100*val_positive/val_total:>5.2f}%)")
    print(f"Background pixels (negative): {val_negative:>15,} ({100*val_negative/val_total:>5.2f}%)")
    print(f"Class imbalance ratio:        {val_negative/val_positive:>15.1f}:1")
    print(f"\nSlices with tumor:            {val_slices_with_tumor:>15,} / {val_total_slices:,} ({100*val_slices_with_tumor/val_total_slices:.1f}%)")
    print(f"Empty slices (no tumor):      {val_total_slices - val_slices_with_tumor:>15,} ({100*(val_total_slices-val_slices_with_tumor)/val_total_slices:.1f}%)")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    pos_weight = train_negative / train_positive
    if pos_weight < 5:
        print("⚠️  WARNING: Very low class imbalance (<5:1)")
        print("    → Check if mask binarization is correct")
        print("    → Verify label files are segmentation masks, not other modalities")
    elif pos_weight > 500:
        print("⚠️  WARNING: Extreme class imbalance (>500:1)")
        print("    → Consider filtering out empty slices")
        print("    → May need aggressive data augmentation")
    else:
        print(f"✓ Class imbalance is typical for medical segmentation ({pos_weight:.1f}:1)")
    
    empty_pct = 100 * (train_total_slices - train_slices_with_tumor) / train_total_slices
    if empty_pct > 80:
        print(f"\n⚠️  WARNING: {empty_pct:.1f}% of slices have NO tumor")
        print("    → Consider filtering empty slices to speed up training")
        print("    → Or implement weighted sampling to focus on tumor slices")
    elif empty_pct > 50:
        print(f"\nℹ️  NOTE: {empty_pct:.1f}% of slices have no tumor (typical for brain scans)")
    else:
        print(f"\n✓ Good distribution: Only {empty_pct:.1f}% empty slices")
    
    print(f"\n{'='*70}\n")
    
    return {
        'train_pos_weight': train_negative / train_positive,
        'val_pos_weight': val_negative / val_positive if val_positive > 0 else float('inf'),
        'train_tumor_pct': 100 * train_positive / train_total,
        'val_tumor_pct': 100 * val_positive / val_total,
        'train_empty_slice_pct': 100 * (train_total_slices - train_slices_with_tumor) / train_total_slices,
        'val_empty_slice_pct': 100 * (val_total_slices - val_slices_with_tumor) / val_total_slices,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze glioma dataset class distribution')
    parser.add_argument('--orientation', type=str, default='axial', 
                        choices=['axial', 'coronal', 'sagittal'],
                        help='Image orientation to analyze')
    parser.add_argument('--batches', type=int, default=100,
                        help='Number of batches to analyze (default: 100)')
    
    args = parser.parse_args()
    
    analyze_class_distribution(orientation=args.orientation, num_batches=args.batches)
