#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p train_logs

CONFIGS=(
  "config/m4d/d2ls_hr.py"
  "config/m4d/d2ls_hr_weighted.py"
  "config/m4d/d2ls_hr_weighted_ctrlow.py"
)

for CONFIG in "${CONFIGS[@]}"; do
  NAME="$(basename "${CONFIG%.py}")"
  LOG_FILE="train_logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date '+%F %T')] start ${CONFIG}"
  python train.py -c "$CONFIG" 2>&1 | tee "$LOG_FILE"
  echo "[$(date '+%F %T')] finish ${CONFIG}"
done

echo "[$(date '+%F %T')] all experiments finished"
