#!/usr/bin/env bash
# Calibration A/B/C: train U-Net under three loss/pos_weight conditions,
# each a full 20-epoch x 3-orientation run with its own timestamp.
set -u
cd /home/sg624ew/glioma
PY=.venv/bin/python
STAMP=$(date +%Y%m%d-%H%M%S)
SUMMARY="training_logs/calibration_ab_${STAMP}.txt"
echo "Calibration A/B/C run started ${STAMP}" | tee "$SUMMARY"

run() {
  local name="$1"; shift
  local log="training_logs/calib_${name}_${STAMP}.log"
  echo "" | tee -a "$SUMMARY"
  echo "=== CONDITION: ${name} | args: $* | $(date +%H:%M:%S) ===" | tee -a "$SUMMARY"
  $PY -u -m src.training_manager "$@" > "$log" 2>&1
  # The newest unet axial checkpoint belongs to the run we just finished.
  local ts
  ts=$(ls -t saved_models/best_axial_*.pt 2>/dev/null | head -1 | sed -E 's#.*best_axial_(.*)\.pt#\1#')
  echo "${name} DONE -> timestamp=${ts} | log=${log} | $(date +%H:%M:%S)" | tee -a "$SUMMARY"
}

run baseline_w96 --model-arch unet --loss bce                 --epochs 20
run bce_w15      --model-arch unet --loss bce --pos-weight 15  --epochs 20
run dice         --model-arch unet --loss dice                --epochs 20

echo "" | tee -a "$SUMMARY"
echo "ALL CONDITIONS DONE $(date +%H:%M:%S)" | tee -a "$SUMMARY"
