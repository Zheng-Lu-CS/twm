#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT/logs/setup_zhenglu_twm_${STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

ENV_NAME="zhenglu_twm"
PYTHON_VERSION="3.10"
PIP_INDEX_URL_DEFAULT="https://pypi.tuna.tsinghua.edu.cn/simple"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

export WANDB_MODE=disabled
export WANDB_DISABLE_GIT=true
export WANDB_DISABLE_CODE=true
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1

echo "== TWM H100 setup =="
echo "Root: $ROOT"
echo "Log:  $LOG_FILE"
echo "Time: $(date -Is)"

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

    echo "ERROR: conda was not found. Install Miniconda/Anaconda first."
    exit 1
}

lib_exists() {
    local lib="$1"
    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "$lib"; then
        return 0
    fi
    find /lib /usr/lib /usr/local/lib -name "$lib" -print -quit 2>/dev/null | grep -q .
}

check_system_deps() {
    local missing=()
    for cmd in ffmpeg xvfb-run; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing+=("$cmd")
        fi
    done

    for lib in libGL.so.1 libSDL2-2.0.so.0 libXrender.so.1 libSM.so.6 libICE.so.6; do
        if ! lib_exists "$lib"; then
            missing+=("$lib")
        fi
    done

    if (( ${#missing[@]} == 0 )); then
        echo "System dependency check: OK"
        return
    fi

    echo "Missing system dependencies: ${missing[*]}"
    local apt_cmd=(apt-get install -y ffmpeg xvfb xauth libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libegl1 libgles2 libglx0 libsdl2-2.0-0)

    if [[ "${INSTALL_SYS_DEPS:-0}" == "1" ]]; then
        if [[ "$(id -u)" == "0" ]]; then
            apt-get update
            "${apt_cmd[@]}"
        elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
            sudo apt-get update
            sudo "${apt_cmd[@]}"
        else
            echo "ERROR: INSTALL_SYS_DEPS=1 was set, but current user cannot run apt-get as root/sudo."
            echo "Run manually: sudo apt-get update && sudo ${apt_cmd[*]}"
            exit 1
        fi
    else
        echo "INSTALL_SYS_DEPS is not 1, so apt install is skipped."
        echo "Suggested command:"
        echo "  sudo apt-get update && sudo ${apt_cmd[*]}"
    fi
}

check_system_deps

load_conda

echo "== Conda =="
conda info

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Updating existing conda environment: $ENV_NAME"
    conda install -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
else
    echo "Creating conda environment: $ENV_NAME"
    conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
fi

conda activate "$ENV_NAME"
python -VV
python -m pip config set global.index-url "$PIP_INDEX_URL_DEFAULT"
python -m pip install --upgrade pip wheel --index-url "$PIP_INDEX_URL_DEFAULT"
# wandb==0.12.21 imports pkg_resources; keep setuptools on a version that still provides it.
python -m pip install setuptools==65.5.1 --index-url "$PIP_INDEX_URL_DEFAULT"

echo "== Installing PyTorch for H100 =="
echo "Using PyTorch CUDA wheel index: $PYTORCH_INDEX_URL"
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "$PYTORCH_INDEX_URL"

echo "== Installing TWM runtime dependencies =="
python -m pip install \
    ale_py==0.8.0 \
    gym==0.26.0 \
    numpy==1.23.1 \
    Pillow==9.2.0 \
    pygame==2.1.0 \
    wandb==0.12.21 \
    setuptools==65.5.1 \
    opencv-python-headless==4.11.0.86 \
    "AutoROM[accept-rom-license]" \
    packaging \
    psutil \
    --index-url "$PIP_INDEX_URL_DEFAULT"

echo "== Installing Atari ROMs =="
if ! AutoROM --accept-license; then
    echo "ERROR: AutoROM failed. Atari ROMs may be missing."
    echo "Manual follow-up: activate '$ENV_NAME' and run 'AutoROM --accept-license' on a network that can reach the ROM source."
    exit 1
fi

echo "== Running environment check =="
"$ROOT/scripts/check_zhenglu_twm_env.sh"

cat <<'EOF'

Setup finished. Useful commands:
  conda activate zhenglu_twm
  ./scripts/check_zhenglu_twm_env.sh
  ./scripts/run_4games_2h100.sh
EOF
