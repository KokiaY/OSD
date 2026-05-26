#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="logs_ablation_m4d"
mkdir -p "$LOG_DIR"

if [[ "${M4D_ABLATION_CHILD:-0}" != "1" ]]; then
  export M4D_ABLATION_CHILD=1
  RUN_LOG="$LOG_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"
  nohup bash "$0" "$@" > "$RUN_LOG" 2>&1 &
  echo "M4D ablation training started in background."
  echo "PID: $!"
  echo "Main log: $RUN_LOG"
  echo "Watch: tail -f $RUN_LOG"
  exit 0
fi

CONFIGS=(
  "config/m4d/ablations/d2ls_swin_v4_abl_full.py"
  "config/m4d/ablations/d2ls_swin_v4_abl_no_mlfa.py"
  "config/m4d/ablations/d2ls_swin_v4_abl_no_cssp.py"
  "config/m4d/ablations/d2ls_swin_v4_abl_no_pfid.py"
  "config/m4d/ablations/d2ls_swin_v4_abl_k3.py"
  "config/m4d/ablations/d2ls_swin_v4_abl_k4.py"
  "config/m4d/ablations/d2ls_swin_v4_abl_l1.py"
  "config/m4d/ablations/d2ls_swin_v4_abl_l2.py"
  "config/m4d/ablations/d2ls_swin_v4_abl_l4.py"
)

echo "Start M4D ablation training at $(date)"
echo "Total runs: ${#CONFIGS[@]}"
START_INDEX="${START_INDEX:-1}"
if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]] || (( START_INDEX < 1 || START_INDEX > ${#CONFIGS[@]} )); then
  echo "Invalid START_INDEX=$START_INDEX. It must be between 1 and ${#CONFIGS[@]}."
  exit 1
fi
echo "Start index: $START_INDEX"

for INDEX in "${!CONFIGS[@]}"; do
  RUN_INDEX=$((INDEX + 1))
  if (( RUN_INDEX < START_INDEX )); then
    echo "Skip [$RUN_INDEX/${#CONFIGS[@]}]: ${CONFIGS[$INDEX]}"
    continue
  fi
  CFG="${CONFIGS[$INDEX]}"
  NAME="$(basename "$CFG" .py)"
  LOG_FILE="$LOG_DIR/${NAME}_$(date +%Y%m%d_%H%M%S).log"
  echo "============================================================"
  echo "Training [$RUN_INDEX/${#CONFIGS[@]}]: $CFG"
  echo "Log: $LOG_FILE"
  echo "Start: $(date)"
  python train.py -c "$CFG" 2>&1 | tee "$LOG_FILE"
  echo "Finish: $(date)"
done

echo "All M4D ablation training finished at $(date)"
