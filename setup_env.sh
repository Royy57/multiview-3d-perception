#!/usr/bin/env bash
# Creates .venv with Python 3.11 and the exact package versions this project is tested with.
#
#   ./setup_env.sh          # NVIDIA GPU (CUDA 12.4 wheels)
#   ./setup_env.sh cpu      # no GPU
set -euo pipefail
CUDA="${1:-cu124}"
PYTHON="${PYTHON:-python3.11}"

[ -d .venv ] || "$PYTHON" -m venv .venv
PY=.venv/bin/python

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements-torch.txt --index-url "https://download.pytorch.org/whl/${CUDA}"
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e . --no-deps

"$PY" -c "import torch, pycolmap, open3d, ultralytics; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('pycolmap', pycolmap.__version__, '| open3d', open3d.__version__, '| ultralytics', ultralytics.__version__)"
echo "Done. Activate with:  source .venv/bin/activate"
