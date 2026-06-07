#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/logs"
SEED="${SEED:-0}"
GPU_IDS="${GPU_IDS:-0,1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TOP_LOG="$ROOT/logs/run_4games_2h100_seed${SEED}_${STAMP}.log"
exec > >(tee -a "$TOP_LOG") 2>&1

echo "== TWM 4-game 2-H100 launcher =="
echo "Root:    $ROOT"
echo "Seed:    $SEED"
echo "GPU_IDS: $GPU_IDS"
echo "Log:     $TOP_LOG"
echo "Time:    $(date -Is)"

IFS=',' read -r -a RAW_GPUS <<< "$GPU_IDS"
GPUS=()
for gpu in "${RAW_GPUS[@]}"; do
    gpu="${gpu#"${gpu%%[![:space:]]*}"}"
    gpu="${gpu%"${gpu##*[![:space:]]}"}"
    if [[ -n "$gpu" ]]; then
        GPUS+=("$gpu")
    fi
done

if (( ${#GPUS[@]} < 2 )); then
    echo "ERROR: GPU_IDS must contain at least two comma-separated GPU ids, for example GPU_IDS=\"0,1\"."
    exit 2
fi

RUN_ONE="$ROOT/scripts/run_one_twm.sh"
if [[ ! -x "$RUN_ONE" ]]; then
    echo "ERROR: $RUN_ONE is not executable. Run: chmod +x scripts/*.sh"
    exit 2
fi

TRAIN_LOGS=()

launch_job() {
    local game="$1"
    local gpu="$2"
    local gpu_label
    gpu_label="$(printf '%s' "$gpu" | sed 's/[^A-Za-z0-9._-]/_/g')"
    local log="$ROOT/logs/train_${game}_seed${SEED}_gpu${gpu_label}_${STAMP}.log"
    TRAIN_LOGS+=("$log")
    TWM_LOG_FILE="$log" "$RUN_ONE" "$game" "$gpu" "$SEED" >/dev/null 2>&1 &
    local pid=$!
    echo "Started: pid=$pid game=$game gpu=$gpu log=$log"
    LAST_PID="$pid"
}

wait_wave() {
    local wave_name="$1"
    shift
    local failed=0
    local pid
    echo "Waiting for $wave_name..."
    for pid in "$@"; do
        if wait "$pid"; then
            echo "$wave_name pid=$pid finished OK"
        else
            local rc=$?
            echo "ERROR: $wave_name pid=$pid failed with exit code $rc"
            failed=1
        fi
    done
    return "$failed"
}

print_failed_log_tails() {
    echo
    echo "Recent training log tails:"
    local log
    for log in "${TRAIN_LOGS[@]}"; do
        echo
        echo "===== $log ====="
        if [[ -f "$log" ]]; then
            tail -n 120 "$log"
        else
            echo "Log file not found."
        fi
    done
}

echo
echo "== Wave 1: Breakout / Boxing =="
launch_job Breakout "${GPUS[0]}"
PID_BREAKOUT="$LAST_PID"
launch_job Boxing "${GPUS[1]}"
PID_BOXING="$LAST_PID"

if ! wait_wave "wave1" "$PID_BREAKOUT" "$PID_BOXING"; then
    echo "Wave 1 failed; Wave 2 will not be started."
    echo "Training logs:"
    printf '  %s\n' "${TRAIN_LOGS[@]}"
    print_failed_log_tails
    exit 1
fi

echo
echo "== Wave 2: Seaquest / RoadRunner =="
launch_job Seaquest "${GPUS[0]}"
PID_SEAQUEST="$LAST_PID"
launch_job RoadRunner "${GPUS[1]}"
PID_ROADRUNNER="$LAST_PID"

if ! wait_wave "wave2" "$PID_SEAQUEST" "$PID_ROADRUNNER"; then
    echo "Wave 2 failed."
    echo "Training logs:"
    printf '  %s\n' "${TRAIN_LOGS[@]}"
    print_failed_log_tails
    exit 1
fi

echo
echo "OK: all four TWM training jobs finished."
echo "Training logs:"
printf '  %s\n' "${TRAIN_LOGS[@]}"
