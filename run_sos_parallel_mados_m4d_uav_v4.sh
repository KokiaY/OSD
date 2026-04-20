#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$ROOT_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$ROOT_DIR"

mkdir -p train_logs

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

SOS_PALSAR_CONFIG="${SOS_PALSAR_CONFIG:-config/sos/d2ls_swin_weighted_palsar_v4.py}"
SOS_SENTINEL_CONFIG="${SOS_SENTINEL_CONFIG:-config/sos/d2ls_swin_weighted_sentinel_v4.py}"
MADOS_CONFIG="${MADOS_CONFIG:-config/mados/d2ls_swin_weighted_v4.py}"
M4D_CONFIG="${M4D_CONFIG:-config/m4d/d2ls_swin_v4.py}"
UAVRGB_CONFIG="${UAVRGB_CONFIG:-config/uavrgb/d2ls_swin_v4.py}"

LAUNCH_LOG="train_logs/sos_parallel_mados_m4d_uav_v4_${RUN_ID}.jobs"
MASTER_LOG="train_logs/sos_parallel_mados_m4d_uav_v4_${RUN_ID}.log"
touch "$LAUNCH_LOG"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LAUNCH_LOG"
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

check_gpu_id() {
  local gpu_id="$1"
  local gpu_count="$2"
  if [[ "$gpu_id" =~ ^[0-9]+$ ]] && (( gpu_id >= gpu_count )); then
    log "warning: GPU ${gpu_id} is outside detected range 0..$((gpu_count - 1))"
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

launch_parallel_job() {
  local dataset_name="$1"
  local config_path="$2"
  local config_name
  local log_file
  local pid_file
  local start_ts_file

  config_name="$(basename "${config_path%.py}")"
  log_file="train_logs/${dataset_name}_${config_name}_${RUN_ID}.log"
  pid_file="train_logs/${dataset_name}_${config_name}_${RUN_ID}.pid"
  start_ts_file="train_logs/${dataset_name}_${config_name}_${RUN_ID}.start"

  log "start ${dataset_name} | config=${config_path} | gpu=${GPU_ID} | log=${log_file} | mode=parallel"
  date +%s > "$start_ts_file"
  (
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train.py -c "$config_path" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    echo "$exit_code" > "${pid_file}.exit"
  ) &
  echo $! > "$pid_file"
}

wait_parallel_job() {
  local dataset_name="$1"
  local config_path="$2"
  local config_name
  local pid_file
  local start_ts_file
  local end_ts
  local duration_seconds
  local exit_code

  config_name="$(basename "${config_path%.py}")"
  pid_file="train_logs/${dataset_name}_${config_name}_${RUN_ID}.pid"
  start_ts_file="train_logs/${dataset_name}_${config_name}_${RUN_ID}.start"

  if [[ ! -f "$pid_file" ]] || [[ ! -f "$start_ts_file" ]]; then
    log "failed ${dataset_name} | missing pid or start marker"
    return 1
  fi

  local pid
  pid="$(cat "$pid_file")"
  wait "$pid"
  exit_code=0
  if [[ -f "${pid_file}.exit" ]]; then
    exit_code="$(cat "${pid_file}.exit")"
  fi

  end_ts="$(date +%s)"
  duration_seconds=$((end_ts - $(cat "$start_ts_file")))
  if (( exit_code != 0 )); then
    log "failed ${dataset_name} | exit_code=${exit_code} | duration=$(format_duration "$duration_seconds")"
    return "$exit_code"
  fi
  log "finish ${dataset_name} | duration=$(format_duration "$duration_seconds")"
}

run_parallel_sos() {
  local palsar_status=0
  local sentinel_status=0

  launch_parallel_job "sos_palsar" "$SOS_PALSAR_CONFIG"
  launch_parallel_job "sos_sentinel" "$SOS_SENTINEL_CONFIG"
  wait_parallel_job "sos_palsar" "$SOS_PALSAR_CONFIG" || palsar_status=$?
  wait_parallel_job "sos_sentinel" "$SOS_SENTINEL_CONFIG" || sentinel_status=$?

  if (( palsar_status != 0 || sentinel_status != 0 )); then
    return 1
  fi
}

run_sequence() {
  local total_start_ts
  local total_end_ts
  local total_duration_seconds

  total_start_ts="$(date +%s)"
  log "sequential training started"
  run_parallel_sos
  run_job "mados" "$MADOS_CONFIG"
  run_job "m4d" "$M4D_CONFIG"
  run_job "uavrgb" "$UAVRGB_CONFIG"
  total_end_ts="$(date +%s)"
  total_duration_seconds=$((total_end_ts - total_start_ts))
  log "all training jobs finished | total_duration=$(format_duration "$total_duration_seconds")"
}

if [[ "${1:-}" == "--run-sequence" ]]; then
  run_sequence
  exit $?
fi

check_config "$SOS_PALSAR_CONFIG"
check_config "$SOS_SENTINEL_CONFIG"
check_config "$MADOS_CONFIG"
check_config "$M4D_CONFIG"
check_config "$UAVRGB_CONFIG"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && (( GPU_COUNT > 0 )); then
    log "detected ${GPU_COUNT} GPU(s)"
    check_gpu_id "$GPU_ID" "$GPU_COUNT"
  fi
fi

log "launching detached runner | gpu=${GPU_ID} | master_log=${MASTER_LOG}"
nohup env \
  RUN_ID="$RUN_ID" \
  PYTHON_BIN="$PYTHON_BIN" \
  SOS_PALSAR_CONFIG="$SOS_PALSAR_CONFIG" \
  SOS_SENTINEL_CONFIG="$SOS_SENTINEL_CONFIG" \
  MADOS_CONFIG="$MADOS_CONFIG" \
  M4D_CONFIG="$M4D_CONFIG" \
  UAVRGB_CONFIG="$UAVRGB_CONFIG" \
  GPU_ID="$GPU_ID" \
  bash "$SCRIPT_PATH" --run-sequence > "$MASTER_LOG" 2>&1 &
PID=$!

log "launched detached runner | pid=${PID}"
log "watch progress with: tail -f ${MASTER_LOG}"
