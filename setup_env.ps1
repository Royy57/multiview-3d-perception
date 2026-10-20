# Creates .venv with Python 3.11 and the exact package versions this project is tested with.
#
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1            # NVIDIA GPU (CUDA 12.4 wheels)
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1 -Cuda cpu  # no GPU
param(
    [string]$Cuda = "cu124",
    [string]$Python = "3.11"
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv with Python $Python"
    py -$Python -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements-torch.txt --index-url "https://download.pytorch.org/whl/$Cuda"
& $py -m pip install -r requirements.txt
& $py -m pip install -e . --no-deps

& $py -c "import torch, pycolmap, open3d, ultralytics; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('pycolmap', pycolmap.__version__, '| open3d', open3d.__version__, '| ultralytics', ultralytics.__version__)"
Write-Host "`nDone. Activate with:  .\.venv\Scripts\Activate.ps1"
