# ABIGAIL Project — Deep Analysis

**Date:** 2026-03-25
**Project:** AI-Boosted Imaging for Glioma Analysis, Identification, and Localization
**TRL Level:** 2–3 (foundational research through experimental validation)
**Partners:** Technical University of Košice (TUKE, lead) and Pavol Jozef Šafárik University (UPJŠ), Slovakia

---

## 1. Project Overview

ABIGAIL is a research project with three specific objectives:

- **SO1** — Glioma classification (G2/G3/G4) from MRI + patient health records using dual-stream multimodal neural networks
- **SO2** — Treatment response assessment (true vs pseudo progression) using CNN/ViT with optional registration
- **SO3** — Tumor and organ-at-risk segmentation using nnU-Net and G-ABIGAIL (generative semi-supervised approach)

**The current codebase focuses exclusively on SO3 — binary tumor segmentation from T2-FLAIR MRI scans.**

---

## 2. Data Assets

### BraTS Dataset (`validated_filtered.csv`)

- **1,145 subjects**, all marked as "Train" (the code creates its own 80/20 split internally)
- **Glioma distribution** — heavily Glioblastoma-dominant:

| Glioma Type | Count | Percentage |
|---|---|---|
| Glioblastoma | 671 | 58.6% |
| Astrocytoma | 244 | 21.3% |
| Oligodendroglioma | 230 | 20.1% |

- **Demographics**: Mean age 52.3 (range 16–83), 56.4% male, 43.6% female
- **Sites**: Duke (39.6%), UCSF (36.5%), UCSD (13.9%), Missouri (7.3%), Indiana (2.7%)
- **Scanners**: GE (64.9%) and Siemens (34.7%); 3T field (76%) and 1.5T (24%)
- **Volume shape**: 182×218×182 voxels (standardized BraTS format, isotropic 1mm)
- **Modality used for training**: Only T2-FLAIR (`-t2f.nii.gz`) + segmentation masks (`-seg.nii.gz`)

### ABG Examples (`data/ABG_examples/`)

5 real patient cases from local hospital data (UNLP Košice):

| Case ID | Type | Modalities | Format |
|---|---|---|---|
| ABG-0001-AC2 | AC2 | T1, T1c, T2, FLAIR, ADC, DWI, Segmentation | NRRD |
| ABG-0015-AC4 | AC4 | Multiple MRI sequences | NRRD |
| ABG-0029-MTS | MTS | Various MRI sequences | NRRD |
| ABG-0064-MTS | MTS | T1c, T1, T2, FLAIR, ADC, DWI, Segmentation | NRRD |
| AC2-0003 | AC2 | Additional example case | NRRD |

Key differences from BraTS:

- **Format**: NRRD (not NIfTI)
- **Non-isotropic voxel spacing** (e.g., 0.83×0.77×5.57mm for ABG-0064 vs 1mm isotropic in BraTS)
- **Different segmentation labels**: BraTS uses {0, 2, 4}; ABG uses {0, 1, 2}
- **Variable dimensions** (e.g., 256×256×35, 288×288×23 — not standardized)
- **Not integrated into the training pipeline** — exist only for visualization/exploration in notebooks

### Filtered Dataset

- `filter_dataset.sh` copies T2-FLAIR + seg files to `data/filtered_dataset/` (~130 cases, 7.8 GB)
- `data_manager.py` points to `/home/sg624ew/glioma_data/filtered_dataset` (a remote SSD path), indicating training was done on a different machine

---

## 3. Model Architecture

Three architectures are implemented:

| Architecture | File | Description |
|---|---|---|
| **UNet** | `src/model.py:64` | Classic encoder-decoder, 4 down/up blocks (64→1024 channels), `conv_transpose` upsampling |
| **DeepLabV3** | `src/model.py:7` | ResNet-101 backbone from torchvision, adapted for 1-channel input |
| **NestedUNet (UNet++)** | `src/unet_nested.py:29` | Dense skip connections, 5 filter levels (64→1024), bilinear upsampling |

**Current training uses only UNet** (`model_arch = 'unet'` in `training_manager.py:74`).

### Key Design Choices

- **Binary segmentation only** — all non-zero mask labels collapsed to 1 via `binarize_mask()` in `src/utils.py`. Multi-class tumor sub-regions (necrotic core, enhancing tumor, edema) are not predicted separately.
- **2D slice-by-slice approach** — 3D volumes are sliced into 2D samples (`BraTS2DDataset`), not using 3D convolutions.
- **Single modality** — only T2-FLAIR, despite BraTS data having T1, T1c, T2w, T2-FLAIR available.
- **Three orientations trained separately** — axial, coronal, and sagittal each get their own model instance.
- **Ensemble** via probability averaging across the three orientation-specific models at inference time.

---

## 4. Training Pipeline

Configuration from `src/training_manager.py`:

| Parameter | Value |
|---|---|
| Loss | BCEWithLogitsLoss with dynamic `pos_weight` |
| Optimizer | Adam, LR=1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6) |
| Gradient clipping | max_norm=1.0 |
| Batch size | 32 |
| Epochs | 20 per orientation |
| Data split | 80/20 at patient level (seed=42) |
| Metrics | IoU (Jaccard), Precision, Recall |
| Best model saving | Based on validation IoU |
| Normalization | Per-slice z-score normalization |
| Padding | Spatial dims padded to be divisible by 16 |

Data is preloaded entirely into RAM with a global volume cache for maximum training speed.

---

## 5. Training Results (November 2025 Run)

### Final Epoch Metrics

| Orientation | Val IoU | Val Precision | Val Recall | Train IoU | Val BCE |
|---|---|---|---|---|---|
| **Axial** | 0.787 | 0.821 | 0.950 | 0.830 | 0.368 |
| **Coronal** | 0.809 | 0.848 | 0.946 | 0.855 | 0.425 |
| **Sagittal** | 0.787 | 0.826 | 0.944 | 0.827 | 0.305 |
| **Average** | **0.795** | **0.832** | **0.947** | — | 0.366 |

### Inference Results (Validation Set)

| Orientation | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| Axial | 0.787 | 0.821 | 0.950 | 0.881 |
| Coronal | 0.809 | 0.848 | 0.946 | 0.895 |
| Sagittal | 0.787 | 0.826 | 0.944 | 0.881 |
| **Average** | **0.795** | **0.832** | **0.947** | **0.886** |

### Single-Case Evaluation (BraTS-GLI-02128-100)

| Method | Dice Score |
|---|---|
| Axial model | 0.934 |
| Coronal model | 0.942 |
| Sagittal model | 0.941 |
| **Ensemble (avg)** | **0.949** |

### Key Observations

1. **Significant overfitting**: Train IoU ~0.83–0.86 vs Val IoU ~0.79. Val BCE increased dramatically throughout training (from ~0.06 at epoch 1 to ~0.37 at epoch 20) while train BCE dropped to ~0.009.
2. **High recall, lower precision**: The model finds most tumor voxels (~95%) but has a ~17% false positive rate, likely from the high `pos_weight` over-compensating for class imbalance.
3. **Coronal orientation performs best** with 0.809 Val IoU.
4. **Ensemble adds value**: The three-orientation ensemble achieves Dice 0.949 on the evaluated case vs individual models at 0.934–0.942.

---

## 6. Notebooks

| Notebook | Purpose | Status |
|---|---|---|
| `examine_single_scan.ipynb` | Orthogonal slice visualization + 3D mesh of BraTS data | Working |
| `examine_masks.ipynb` | Slice overlay + label analysis | Working; found labels {0, 2, 4} in BraTS |
| `examine_ABG_scan.ipynb` | ABG case exploration with multi-modality + 3D mesh | Working; NRRD loading, voxel spacing correction, brain extraction |
| `examine_ABG_scan_ABG-0001-AC2.ipynb` | Case-specific ABG analysis (ABG-0001-AC2) | Available |
| `examine_ABG_scan_ABG-0064-MTS.ipynb` | Case-specific ABG analysis (ABG-0064-MTS) | Available |
| `ABG_analysis.ipynb` | Comprehensive ABG dataset analysis (6 sections: structure, visualization, intensity stats, volume sizes, segmentation analysis, summary) | Working |
| `model_predictions.ipynb` | End-to-end inference + ensemble + 3D comparison meshes | Working; Dice scores and Plotly 3D visualizations generated |

Supporting documentation: `visualization/MRI_VISUALIZATION_EXPLAINED.md` — comprehensive guide to MRI brain visualization, voxel spacing, and brain extraction pipeline.

---

## 7. Source Code Structure

```
src/
├── main.py              # Entry point — builds data pairs, verifies one batch
├── model.py             # UNet, DeepLabV3, imports NestedUNet
├── unet_nested.py       # UNet++ (NestedUNet) architecture
├── dataset.py           # BraTS2DDataset — 3D→2D slicing, preloading, normalization
├── data_manager.py      # Data loading, train/val split, DataLoader creation
├── training_manager.py  # Full training loop (3 orientations × 20 epochs)
├── inference.py         # Load checkpoints, evaluate on validation set
├── predict.py           # Single-case prediction, ensemble, Dice calculation
├── diagnose_data.py     # Class distribution analysis and recommendations
├── utils.py             # binarize_mask() utility
└── __init__.py          # Package init
```

---

## 8. Critical Gaps and Issues

### Architecture / Training

1. **Only T2-FLAIR used** — BraTS provides 4 modalities (T1, T1c, T2w, T2-FLAIR). Multi-channel input would significantly improve segmentation, especially for enhancing tumor regions visible on T1c.
2. **2D slicing loses 3D context** — the project description mentions nnU-Net (3D), but implementation is 2D. The three-orientation ensemble partially compensates, but native 3D convolutions would capture volumetric context better.
3. **Binary-only segmentation** — collapsing labels {1, 2, 4} → {0, 1} loses clinically important sub-region information (enhancing tumor, peritumoral edema, necrotic core).
4. **Heavy overfitting** — validation loss diverges strongly from training loss by epoch 20. Needs: data augmentation, dropout, weight decay, early stopping, or fewer epochs.
5. **No data augmentation** — `transforms=None` is always passed to the dataset. Random flips, rotations, intensity shifts, and elastic deformation are standard in medical image segmentation.
6. **Hardcoded paths** — `data_manager.py` points to `/home/sg624ew/glioma_data/filtered_dataset`, making the code non-portable.

### Data Integration

7. **ABG examples not used for training or evaluation** — they are in NRRD format with different dimensions, spacing, and label encoding. No preprocessing pipeline exists to convert them to BraTS-compatible format.
8. **No held-out test set** — all 1,145 CSV rows are labeled "Train". The 80/20 split is internal; there is no independent test set for final evaluation.
9. **CSV Train/Test/Validation column is always "Train"** — the data manager's filter for "Train" rows is effectively a no-op.

### Missing Project Objectives

10. **SO1 (Classification) — not started**: No classification head, no patient metadata integration, no dual-stream architecture, no NLP module for clinical text.
11. **SO2 (Treatment Response) — not started**: No registration module (VoxelMorph), no progression vs pseudo-progression classification.
12. **Advanced methods not implemented**: No multi-task learning, no synthetic data generation (CGANs/Stable Diffusion), no semi-supervised learning, no G-ABIGAIL generative approach, no explainability (LIME/SHAP).

---

## 9. Summary

The project is at an **early prototype stage for SO3 only**. A working 2D UNet binary segmentation pipeline exists with:

- Functional data loading, training, inference, and visualization
- Reasonable validation IoU (~0.79) but significant overfitting
- Strong single-case ensemble Dice (~0.95)
- Rich visualization work (orthogonal slices, 3D Plotly meshes, multi-modality comparisons)
- ABG hospital data explored in notebooks but not integrated into the ML pipeline

The gap between the project description (multi-task learning, GANs, VoxelMorph, semi-supervised learning, G-ABIGAIL, dual-stream multimodal networks) and the current implementation (single-modality 2D binary UNet) is substantial.

### Recommended Next Steps (Priority Order)

1. **Address overfitting** — add data augmentation (flips, rotations, elastic deformation), early stopping, and weight decay
2. **Multi-modal input** — use all 4 MRI channels (T1, T1c, T2w, T2-FLAIR) as a 4-channel input
3. **Multi-class segmentation** — predict tumor sub-regions {1, 2, 4} separately instead of binary
4. **3D architecture** — implement nnU-Net or 3D UNet for volumetric context
5. **ABG data pipeline** — create preprocessing to convert NRRD → NIfTI, resample to common spacing, and map labels for training/evaluation on hospital data
6. **Begin SO1/SO2** — classification head and treatment response modules
