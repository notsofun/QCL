# Cloud Deployment

This project is a Python/PyTorch research codebase for EEG-image and video-text contrastive learning with PennyLane quantum layers.

## Recommended Server

- Ubuntu 20.04/22.04
- Python 3.10
- NVIDIA driver compatible with CUDA 12.1 for GPU training
- At least 16 GB RAM; GPU memory depends on batch size and feature backend

## One-Command Setup

From the repository root on the cloud server:

```bash
bash scripts/setup_cloud.sh
```

The script installs common system packages, creates a Python environment, installs PyTorch, installs `requirements.txt`, and verifies the important imports.

Useful options:

```bash
# CPU-only environment
CUDA_VERSION=cpu bash scripts/setup_cloud.sh

# Custom environment name
ENV_NAME=qcl-prod bash scripts/setup_cloud.sh
```

## Conda Alternative

If you prefer Conda/Mamba:

```bash
conda env create -f environment.yml
conda activate qcl
```

## Smoke Tests

Check the Python dependencies:

```bash
python - <<'PY'
import torch
import pennylane
import transformers
import torch_geometric
print("cuda:", torch.cuda.is_available())
print("torch:", torch.__version__)
PY
```

Run a tiny video-text experiment using the included smoke manifest:

```bash
python model/qcl_train_refactored.py \
  --task video_text \
  --manifest Data/qcl-video-text-smoke/manifest.csv \
  --epoch 1 \
  --batch-size 2 \
  --feature_backend handcraft \
  --adapter raw
```

## Data Notes

- EEG-image training expects `Data/Things-EEG2/...` as described in `README.md`.
- Video-text training expects a manifest CSV with video and text/caption columns.
- Hugging Face CLIP/XCLIP backends download model weights on first use, so the server needs network access or a pre-populated model cache.
