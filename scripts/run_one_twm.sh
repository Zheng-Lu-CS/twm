#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GAME="${1:-}"
GPU_ID="${2:-}"
SEED="${3:-0}"

if [[ -z "$GAME" || -z "$GPU_ID" ]]; then
    echo "Usage: $0 GAME GPU_ID [SEED]"
    echo "Example: $0 Breakout 0 0"
    exit 2
fi

case "$GAME" in
    Breakout|Boxing|Seaquest|RoadRunner) ;;
    *)
        echo "ERROR: unsupported game '$GAME'. Use Breakout, Boxing, Seaquest, or RoadRunner."
        exit 2
        ;;
esac

mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
GPU_LABEL="$(printf '%s' "$GPU_ID" | sed 's/[^A-Za-z0-9._-]/_/g')"
LOG_FILE="${TWM_LOG_FILE:-$ROOT/logs/train_${GAME}_seed${SEED}_gpu${GPU_LABEL}_${STAMP}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

ENV_NAME="zhenglu_twm"

export WANDB_MODE=disabled
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export PYTHONUNBUFFERED=1

load_conda() {
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base)"
        # shellcheck disable=SC1091
        source "$conda_base/etc/profile.d/conda.sh"
        return
    fi

    for conda_sh in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "$conda_sh" ]]; then
            # shellcheck disable=SC1090
            source "$conda_sh"
            return
        fi
    done

    echo "ERROR: conda was not found."
    exit 1
}

echo "== TWM training job =="
echo "Root: $ROOT"
echo "Game: $GAME"
echo "GPU:  $GPU_ID"
echo "Seed: $SEED"
echo "Log:  $LOG_FILE"
echo "Time: $(date -Is)"

if command -v git >/dev/null 2>&1 && [[ -d "$ROOT/.git" ]]; then
    git config --global --add safe.directory "$ROOT" || true
fi

load_conda
conda activate "$ENV_NAME"

echo "== CUDA preflight =="
CUDA_VISIBLE_DEVICES="$GPU_ID" python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("cuda:0 is not available after CUDA_VISIBLE_DEVICES binding")
props = torch.cuda.get_device_properties(0)
print(f"cuda:0 -> {props.name}, capability={props.major}.{props.minor}, memory_gb={props.total_memory / 1024 ** 3:.2f}")
x = torch.ones((128, 128), device="cuda:0")
y = x @ x
torch.cuda.synchronize()
print("preflight matmul:", tuple(y.shape), float(y[0, 0].cpu()))
PY

echo "== Starting official TWM command =="
set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" python -O twm/main.py \
    --game "$GAME" \
    --seed "$SEED" \
    --device cuda:0 \
    --cpu_p 1.0 \
    --wandb disabled
RC=$?
set -e

if (( RC != 0 )); then
    echo "ERROR: training failed with exit code $RC"
    echo "Log: $LOG_FILE"
    exit "$RC"
fi

echo "OK: training finished."
echo "Log: $LOG_FILE"
