#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-qcl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION="${CUDA_VERSION:-cu121}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "[1/5] Installing system packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git curl wget ffmpeg libgl1 libglib2.0-0
else
  echo "apt-get not found; skipping system package installation."
fi

echo "[2/5] Preparing Python environment: ${ENV_NAME}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
  fi
  conda activate "${ENV_NAME}"
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
fi

echo "[3/5] Installing PyTorch for ${CUDA_VERSION}"
if [ "${CUDA_VERSION}" = "cpu" ]; then
  python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<2.5" "torchvision>=0.17,<0.20" "torchaudio>=2.2,<2.5"
else
  python -m pip install --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" "torch>=2.2,<2.5" "torchvision>=0.17,<0.20" "torchaudio>=2.2,<2.5"
fi

echo "[4/5] Installing project dependencies"
python -m pip install -r requirements.txt

echo "[5/5] Verifying key imports"
python - <<'PY'
import importlib

modules = [
    "torch",
    "torch_geometric",
    "pennylane",
    "transformers",
    "cv2",
    "mne",
    "datasets",
    "huggingface_hub",
    "einops",
    "open_clip",
]

for name in modules:
    importlib.import_module(name)
    print(f"ok: {name}")
PY

echo
echo "Environment is ready."
if command -v conda >/dev/null 2>&1; then
  echo "Activate with: conda activate ${ENV_NAME}"
else
  echo "Activate with: source .venv/bin/activate"
fi
