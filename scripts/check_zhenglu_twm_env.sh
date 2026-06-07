#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT/logs/check_zhenglu_twm_${STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

ENV_NAME="zhenglu_twm"
FAIL=0

export WANDB_MODE=disabled
export WANDB_DISABLE_GIT=true
export WANDB_DISABLE_CODE=true
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export PYTHONUNBUFFERED=1

echo "== TWM H100 environment check =="
echo "Root: $ROOT"
echo "Log:  $LOG_FILE"

mark_fail() {
    echo "ERROR: $*"
    FAIL=1
}

run_required() {
    echo
    echo "== $* =="
    "$@"
    local rc=$?
    if (( rc != 0 )); then
        mark_fail "command failed with exit code $rc: $*"
    fi
}

load_conda() {
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base)"
        # shellcheck disable=SC1091
        source "$conda_base/etc/profile.d/conda.sh"
        return 0
    fi

    for conda_sh in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "$conda_sh" ]]; then
            # shellcheck disable=SC1090
            source "$conda_sh"
            return 0
        fi
    done
    return 1
}

if ! load_conda; then
    mark_fail "conda was not found."
else
    if ! conda activate "$ENV_NAME"; then
        mark_fail "could not activate conda environment '$ENV_NAME'."
    fi
fi

run_required hostname
run_required date -Is
run_required uname -a
if [[ -f /etc/os-release ]]; then
    run_required cat /etc/os-release
else
    mark_fail "/etc/os-release not found."
fi
run_required conda info
run_required conda env list
run_required python -VV
run_required python -m pip list
run_required nvidia-smi
run_required df -h / /tmp /dev/shm

echo
echo "== Python package, CUDA, and Atari checks =="
python - <<'PY'
import importlib
import os
import sys

packages = ["torch", "torchvision", "gym", "ale_py", "numpy", "PIL", "pygame", "wandb", "cv2"]
for name in packages:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"{name}: {version}")

import torch

print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.get_arch_list():", torch.cuda.get_arch_list())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. One visible H100/MIG device is enough for this check.")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(
        f"gpu[{i}]: name={props.name}, capability={props.major}.{props.minor}, "
        f"total_memory_gb={props.total_memory / 1024 ** 3:.2f}"
    )

a = torch.randn(1024, 1024, device="cuda:0")
b = torch.randn(1024, 1024, device="cuda:0")
c = a @ b
torch.cuda.synchronize()
print("cuda matmul smoke:", tuple(c.shape), float(c[0, 0].detach().cpu()))

from ale_py.roms.utils import rom_name_to_id

for game in ["Breakout", "Boxing", "Seaquest", "RoadRunner"]:
    rom_id = rom_name_to_id(game)
    print(f"ROM lookup {game}: {rom_id}")

sys.path.insert(0, os.path.join(os.getcwd(), "twm"))
import utils

env = utils.create_atari_env(
    "Breakout",
    noop_max=30,
    frame_skip=4,
    frame_stack=4,
    frame_size=64,
    episodic_lives=True,
    grayscale=True,
    time_limit=27000,
)
obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(
    "Breakout env smoke:",
    "obs_shape=", getattr(obs, "shape", None),
    "reward=", reward,
    "terminated=", terminated,
    "truncated=", truncated,
)
env.close()
PY
PY_RC=$?
if (( PY_RC != 0 )); then
    mark_fail "Python package/CUDA/Atari check failed with exit code $PY_RC."
fi

run_required python -O twm/main.py --help

echo
if (( FAIL != 0 )); then
    echo "TWM H100 environment check FAILED. See log: $LOG_FILE"
    exit 1
fi

echo "OK: TWM H100 environment check passed. See log: $LOG_FILE"
