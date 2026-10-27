param([string]$EnvironmentPath = '.venv')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    uv venv $EnvironmentPath --python 3.13 --allow-existing
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare Python 3.13 environment' }
    $pythonPath = Join-Path $EnvironmentPath 'Scripts/python.exe'
    # Both indexes are trusted explicit sources; permit PyPI to supply packages
    # whose older versions also appear in the PyTorch wheel index.
    uv pip install --python $pythonPath --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-v3-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'V3 dependency installation failed' }
    & $pythonPath -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke test failed' }
} finally {
    Pop-Location
}
