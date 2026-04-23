#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$ROOT_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$ROOT_DIR"

mkdir -p train_logs

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

MASTER_LOG="train_logs/uavrgb_m4d_mados_v4_remaining_ablations_${RUN_ID}.log"
JOBS_LOG="train_logs/uavrgb_m4d_mados_v4_remaining_ablations_${RUN_ID}.jobs"
touch "$JOBS_LOG"

CONFIGS=(
  "uavrgb config/uavrgb/d2ls_swin_v4_reg_none.py"
  "uavrgb config/uavrgb/d2ls_swin_v4_reg_cls.py"
  "uavrgb config/uavrgb/d2ls_swin_v4_reg_div.py"
  "uavrgb config/uavrgb/d2ls_swin_v4_agg_mean.py"
  "uavrgb config/uavrgb/d2ls_swin_v4_agg_max.py"
  "m4d config/m4d/d2ls_swin_v4_reg_none.py"
  "m4d config/m4d/d2ls_swin_v4_reg_cls.py"
  "m4d config/m4d/d2ls_swin_v4_reg_div.py"
  "m4d config/m4d/d2ls_swin_v4_agg_mean.py"
  "m4d config/m4d/d2ls_swin_v4_agg_max.py"
  "mados config/mados/d2ls_swin_weighted_v4_reg_none.py"
  "mados config/mados/d2ls_swin_weighted_v4_reg_cls.py"
  "mados config/mados/d2ls_swin_weighted_v4_reg_div.py"
  "mados config/mados/d2ls_swin_weighted_v4_agg_mean.py"
  "mados config/mados/d2ls_swin_weighted_v4_agg_max.py"
)

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$JOBS_LOG"
}

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02dh:%02dm:%02ds" "$hours" "$minutes" "$seconds"
}

check_config() {
  local config_path="$1"
  if [[ ! -f "$config_path" ]]; then
    log "missing config: $config_path"
    exit 1
  fi
}

run_job() {
  local dataset_name="$1"
  local config_path="$2"
  local config_name
  local log_file
  local exit_code
  local start_ts
  local end_ts
  local duration_seconds

  config_name="$(basename "${config_path%.py}")"
  log_file="train_logs/${dataset_name}_${config_name}_${RUN_ID}.log"
  start_ts="$(date +%s)"

  log "start ${dataset_name} | config=${config_path} | gpu=${GPU_ID} | log=${log_file}"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train.py -c "$config_path" 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e
  end_ts="$(date +%s)"
  duration_seconds=$((end_ts - start_ts))

  if (( exit_code != 0 )); then
    log "failed ${dataset_name} | exit_code=${exit_code} | duration=$(format_duration "$duration_seconds")"
    return "$exit_code"
  fi

  log "finish ${dataset_name} | duration=$(format_duration "$duration_seconds")"
}

run_sequence() {
  local total_start_ts
  local total_end_ts
  local total_duration_seconds
  local entry
  local dataset_name
  local config_path

  total_start_ts="$(date +%s)"
  log "remaining ablation training started"
  for entry in "${CONFIGS[@]}"; do
    dataset_name="${entry%% *}"
    config_path="${entry#* }"
    run_job "$dataset_name" "$config_path"
  done
  total_end_ts="$(date +%s)"
  total_duration_seconds=$((total_end_ts - total_start_ts))
  log "all remaining ablation jobs finished | total_duration=$(format_duration "$total_duration_seconds")"
}

if [[ "${1:-}" == "--run-sequence" ]]; then
  run_sequence
  exit $?
fi

for entry in "${CONFIGS[@]}"; do
  check_config "${entry#* }"
done

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && (( GPU_COUNT > 0 )); then
    log "detected ${GPU_COUNT} GPU(s)"
  fi
fi

log "launching detached sequential runner | gpu=${GPU_ID} | master_log=${MASTER_LOG}"
nohup env \
  RUN_ID="$RUN_ID" \
  PYTHON_BIN="$PYTHON_BIN" \
  GPU_ID="$GPU_ID" \
  bash "$SCRIPT_PATH" --run-sequence > "$MASTER_LOG" 2>&1 &
PID=$!

log "launched sequential runner | pid=${PID}"
log "watch progress with: tail -f ${MASTER_LOG}"
