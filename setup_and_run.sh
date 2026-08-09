#!/usr/bin/env bash
# ============================================================
# setup_and_run.sh — 3dReConstruct one-shot bootstrap
#
# Turns any fresh Linux/macOS machine into a running
# 3dReConstruct server: checks compatibility, installs missing
# dependencies, sets up isolated environments for the backend
# and for FastGS, then launches the FastAPI server so it can
# RECEIVE a video -> PROCESS it through COLMAP+FastGS+cleanup
# -> SEND back the rendered .ply splat.

#Manual check:

# 1. Check the NVIDIA driver is installed and the GPU is visible
#nvidia-smi
# 2. Check the CUDA toolkit version actually installed (nvcc)
#nvcc --version
# 3. Check what PyTorch (inside your venv) actually sees
#source .venv-fastgs/bin/activate
#python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Torch CUDA version:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
#deactivate


# Usage:

#------------------- Case A — fresh Ubuntu box, nothing cloned yet:

#sudo apt-get update -y
#sudo apt-get install -y curl
#curl -O https://raw.githubusercontent.com/adityandandia/3dReConstruct/master/setup_and_run.sh
#chmod +x setup_and_run.sh
#./setup_and_run.sh

#-------------------- Case B — you already git cloned it there:
#cd 3dReConstruct
#chmod +x setup_and_run.sh
#./setup_and_run.sh

#   ./setup_and_run.sh --force      # skip compatibility hard-stops
#
# Safe to re-run — every step checks before acting.
# ============================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[FAIL]${NC} $*"; }
step() { echo -e "\n${BLUE}==>${NC} $*"; }

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# GitHub repo this script belongs to — used to self-clone / self-update.
REPO_URL="https://github.com/adityandandia/3dReConstruct.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10
MIN_RAM_GB=8
MIN_DISK_GB=15
MIN_VRAM_MB=4000

# ============================================================
# STEP -1 — OS / package manager detection (needed early for git)
# ============================================================
step "Detecting OS and package manager"

OS_TYPE="unknown"; PKG_MANAGER="none"; ARCH="$(uname -m)"

if [[ "$(uname -s)" == "Darwin" ]]; then
    OS_TYPE="macos"
    command -v brew >/dev/null 2>&1 && PKG_MANAGER="brew"
elif [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS_TYPE="${ID:-linux}"
    if command -v apt-get >/dev/null 2>&1; then PKG_MANAGER="apt"
    elif command -v dnf >/dev/null 2>&1; then PKG_MANAGER="dnf"
    elif command -v pacman >/dev/null 2>&1; then PKG_MANAGER="pacman"
    fi
fi
log "OS: $OS_TYPE | Arch: $ARCH | Package manager: $PKG_MANAGER"

pkg_install() {  # apt_name dnf_name pacman_name brew_name
    local apt_n=$1 dnf_n=$2 pacman_n=$3 brew_n=$4
    case "$PKG_MANAGER" in
        apt)    sudo apt-get update -qq && sudo apt-get install -y "$apt_n" ;;
        dnf)    sudo dnf install -y "$dnf_n" ;;
        pacman) sudo pacman -Sy --noconfirm "$pacman_n" ;;
        brew)   brew install "$brew_n" ;;
        *)      warn "No supported package manager — cannot auto-install '$apt_n'." ;;
    esac
}

# ============================================================
# STEP 0 — Sync with GitHub (self-clone if standalone, self-update if not)
# ============================================================
step "Syncing with GitHub repo"

if ! command -v git >/dev/null 2>&1; then
    warn "git not found — installing (required to clone/update the repo)..."
    pkg_install git git git git
    command -v git >/dev/null 2>&1 || { err "git install failed — install manually and re-run."; exit 1; }
fi

if [[ -d "$SCRIPT_DIR/.git" ]]; then
    # Script lives inside an existing clone — pull the latest changes.
    log "Existing repo detected at $SCRIPT_DIR — pulling latest"
    git -C "$SCRIPT_DIR" pull --ff-only || warn "git pull failed (local changes / diverged history?) — continuing with current checkout."
    PROJECT_ROOT="$SCRIPT_DIR"
else
    # Script was downloaded standalone (e.g. curl'd on its own) — clone fresh next to it.
    CLONE_DIR="$SCRIPT_DIR/3dReConstruct"
    if [[ -d "$CLONE_DIR/.git" ]]; then
        log "Found existing clone at $CLONE_DIR — pulling latest"
        git -C "$CLONE_DIR" pull --ff-only || warn "git pull failed — continuing with current checkout."
    else
        warn "No repo found next to this script — cloning $REPO_URL"
        git clone "$REPO_URL" "$CLONE_DIR"
    fi
    PROJECT_ROOT="$CLONE_DIR"
fi

log "Using project at: $PROJECT_ROOT"

BACKEND_DIR="$PROJECT_ROOT/backend"
FASTGS_DIR="$PROJECT_ROOT/FastGS"
VENV_DIR="$PROJECT_ROOT/.venv"
FASTGS_VENV_DIR="$PROJECT_ROOT/.venv-fastgs"
ENV_FILE="$PROJECT_ROOT/.env.generated"

# ============================================================
# STEP 1 — Compatibility check (hardware/OS)
# ============================================================
step "Running compatibility check"

COMPAT_FAIL=0

if [[ "$ARCH" != "x86_64" && "$ARCH" != "amd64" ]]; then
    warn "Architecture is '$ARCH' — COLMAP/FastGS prebuilt CUDA kernels target x86_64. May need source builds."
    COMPAT_FAIL=1
fi

# RAM check
if [[ "$OS_TYPE" == "macos" ]]; then
    RAM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
else
    RAM_GB=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
fi
if (( RAM_GB < MIN_RAM_GB )); then
    warn "System RAM is ${RAM_GB}GB (recommended >= ${MIN_RAM_GB}GB for COLMAP dense reconstruction)."
    COMPAT_FAIL=1
else
    log "RAM: ${RAM_GB}GB"
fi

# Disk check
DISK_AVAIL_GB=$(( $(df -Pk "$PROJECT_ROOT" | tail -1 | awk '{print $4}') / 1024 / 1024 ))
if (( DISK_AVAIL_GB < MIN_DISK_GB )); then
    warn "Free disk space is ${DISK_AVAIL_GB}GB (recommended >= ${MIN_DISK_GB}GB for frames/dense recon/models)."
    COMPAT_FAIL=1
else
    log "Free disk: ${DISK_AVAIL_GB}GB"
fi

# GPU check
GPU_OK=0
if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    log "GPU: $GPU_NAME (${VRAM_MB}MB VRAM)"
    if (( VRAM_MB < MIN_VRAM_MB )); then
        warn "VRAM (${VRAM_MB}MB) is below the tested minimum (${MIN_VRAM_MB}MB). FastGS training may OOM."
        COMPAT_FAIL=1
    else
        GPU_OK=1
    fi
else
    warn "No NVIDIA GPU detected (nvidia-smi missing). FastGS training requires CUDA — this machine can run"
    warn "COLMAP/backend only, not the splatting step, unless a GPU + driver is added later."
    COMPAT_FAIL=1
fi

# nvcc / CUDA toolkit check — driver presence (nvidia-smi) alone is not
# enough to compile FastGS's custom CUDA rasterizer extensions in Step 5.
if command -v nvcc >/dev/null 2>&1; then
    log "nvcc found: $(nvcc --version | tail -1)"
else
    warn "nvcc not found — CUDA toolkit may be missing (driver alone isn't enough to compile CUDA extensions)."
    COMPAT_FAIL=1
fi

if (( COMPAT_FAIL == 1 )); then
    if (( FORCE == 1 )); then
        warn "Compatibility issues found, continuing anyway (--force)."
    else
        err "Compatibility issues found above. Re-run with --force to proceed anyway,"
        err "or fix the flagged item (more RAM/disk/GPU) first."
        exit 1
    fi
fi

# ============================================================
# STEP 2 — Core system dependencies
# ============================================================
step "Checking core dependencies"

check_or_install() {  # cmd apt dnf pacman brew
    local cmd=$1 apt_pkg=$2 dnf_pkg=$3 pacman_pkg=$4 brew_pkg=$5
    if command -v "$cmd" >/dev/null 2>&1; then
        log "$cmd found: $(command -v "$cmd")"
    else
        warn "$cmd not found — installing..."
        pkg_install "$apt_pkg" "$dnf_pkg" "$pacman_pkg" "$brew_pkg"
        command -v "$cmd" >/dev/null 2>&1 && log "$cmd installed." || { err "$cmd still missing — install manually."; exit 1; }
    fi
}

check_or_install git    git    git    git    git
check_or_install ffmpeg ffmpeg ffmpeg ffmpeg ffmpeg
check_or_install cmake  cmake  cmake  cmake  cmake
check_or_install curl   curl   curl   curl   curl
check_or_install gcc    gcc    gcc    gcc    gcc

if ! command -v colmap >/dev/null 2>&1; then
    warn "colmap not found — installing..."
    pkg_install colmap colmap colmap colmap
    command -v colmap >/dev/null 2>&1 && log "colmap installed." || { err "colmap missing — build from https://colmap.github.io/install.html"; exit 1; }
else
    log "colmap found: $(command -v colmap)"
fi

# ngrok — used by config.py's auto-detection for the phone-facing tunnel
# URL. Not started automatically here (requires an authtoken), just
# checked for and flagged if missing so the operator knows to install it.
if ! command -v ngrok >/dev/null 2>&1; then
    warn "ngrok not found — phone/remote clients won't be able to reach this server"
    warn "unless BACKEND_URL is set to a reachable LAN IP. Install from https://ngrok.com/download"
    warn "and run 'ngrok http $PORT' separately if you need remote access."
else
    log "ngrok found: $(command -v ngrok)"
fi

# ============================================================
# STEP 3 — Python 3.10+
# ============================================================
step "Checking Python (>= ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR})"

PYTHON_BIN=""
for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ "${ver%%.*}" -eq "$PYTHON_MIN_MAJOR" && "${ver##*.}" -ge "$PYTHON_MIN_MINOR" ]]; then
            PYTHON_BIN=$(command -v "$candidate"); log "Using $candidate ($ver)"; break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    warn "Python 3.10+ not found — installing..."
    case "$PKG_MANAGER" in
        apt)
            sudo apt-get update -qq
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa || true
            sudo apt-get update -qq
            sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
            PYTHON_BIN=$(command -v python3.10 || true) ;;
        dnf)    sudo dnf install -y python3.10 ; PYTHON_BIN=$(command -v python3.10 || true) ;;
        pacman) sudo pacman -Sy --noconfirm python ; PYTHON_BIN=$(command -v python3 || true) ;;
        brew)   brew install python@3.10 ; PYTHON_BIN=$(command -v python3.10 || true) ;;
        *)      err "Install Python 3.10+ manually."; exit 1 ;;
    esac
    [[ -z "$PYTHON_BIN" ]] && { err "Python install failed."; exit 1; }
fi

# ============================================================
# STEP 4 — Backend venv + deps
# ============================================================
step "Setting up backend virtual environment"

[[ -d "$VENV_DIR" ]] || "$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

if [[ -f "$BACKEND_DIR/requirements.txt" ]]; then
    pip install -r "$BACKEND_DIR/requirements.txt"
    log "backend/requirements.txt installed"
else
    warn "backend/requirements.txt missing — installing minimal known deps"
    pip install fastapi uvicorn python-multipart numpy scipy scikit-learn requests
fi
deactivate

# splat-transform — external CLI called by clean_splat() at the end of the
# pipeline. Not part of pip/apt; check separately and install via npm if
# Node.js is present, otherwise just warn (tasks.py already tolerates this
# failing, but it fails silently, so flag it here instead).
if ! command -v splat-transform >/dev/null 2>&1; then
    warn "splat-transform not found — final .sog compression step will silently fail."
    if command -v npm >/dev/null 2>&1; then
        warn "Installing splat-transform via npm..."
        npm install -g splat-transform || warn "splat-transform install failed — install manually."
    else
        warn "npm not found — install Node.js, then 'npm install -g splat-transform' manually."
    fi
else
    log "splat-transform found: $(command -v splat-transform)"
fi

# ============================================================
# STEP 5 — FastGS submodule + isolated env
# ============================================================
step "Setting up FastGS"

if [[ ! -d "$FASTGS_DIR" || -z "$(ls -A "$FASTGS_DIR" 2>/dev/null)" ]]; then
    warn "FastGS submodule missing — initializing"
    git -C "$PROJECT_ROOT" submodule update --init --recursive
fi
log "FastGS present at $FASTGS_DIR"

if [[ ! -d "$FASTGS_VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$FASTGS_VENV_DIR"
    log "Created FastGS venv at $FASTGS_VENV_DIR"
fi
# shellcheck disable=SC1091
source "$FASTGS_VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install "numpy<2" -q

if (( GPU_OK == 1 )); then
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q || \
        warn "CUDA-build torch install failed — check CUDA/driver version match."
else
    warn "Skipping CUDA torch install (no compatible GPU detected)."
fi

if [[ -f "$FASTGS_DIR/requirements.txt" ]]; then
    pip install -r "$FASTGS_DIR/requirements.txt" || warn "Some FastGS requirements failed — check CUDA extension build logs."
fi
FASTGS_PYTHON_BIN="$FASTGS_VENV_DIR/bin/python"
deactivate
log "FastGS interpreter: $FASTGS_PYTHON_BIN"

# ============================================================
# STEP 6 — Write resolved paths for tasks.py (env vars)
# ============================================================
step "Writing resolved environment (.env.generated)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ -z "$LAN_IP" ]] && LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "127.0.0.1")

cat > "$ENV_FILE" <<EOF
FASTGS_DIR=$FASTGS_DIR
FASTGS_PYTHON=$FASTGS_PYTHON_BIN
BACKEND_URL=http://$LAN_IP:$PORT
EOF
log "Wrote $ENV_FILE"
warn "Requires the tasks.py portability patch: read FASTGS_DIR/FASTGS_PYTHON from os.environ."

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# ============================================================
# STEP 7 — Launch the server (receive -> process -> send)
# ============================================================
step "Starting backend server"

source "$VENV_DIR/bin/activate"
cd "$BACKEND_DIR"

[[ -z "$LAN_IP" ]] && LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "127.0.0.1")

log "Server will accept video uploads at:  http://$LAN_IP:$PORT/upload"
log "Rendered splat will be served at:     http://$LAN_IP:$PORT/splat/latest.ply"
log "Point the Android app's backend URL to the address above."

exec uvicorn main:app --host "$HOST" --port "$PORT" --reload
