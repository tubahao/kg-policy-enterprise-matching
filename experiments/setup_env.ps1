param(
    [string]$EnvName = ".venv",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment at $EnvName ..."
& $PythonExe -m venv $EnvName

$venvPython = Join-Path $EnvName "Scripts\python.exe"

Write-Host "Upgrading pip ..."
& $venvPython -m pip install --upgrade pip

# Install PyTorch (CUDA 12.1 wheels). Change to 'cpu' wheels if no GPU.
Write-Host "Installing PyTorch (CUDA 12.1)..."
& $venvPython -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Install DGL GPU wheels. For CPU, use: pip install dgl==2.3.0
Write-Host "Installing DGL (CUDA 12.1)..."
& $venvPython -m pip install dgl-cu121==2.3.0 -f https://data.dgl.ai/wheels/cu121/repo.html

# Core requirements (GraphRAG + utilities)
Write-Host "Installing project requirements..."
& $venvPython -m pip install -r env/requirements_graph.txt

Write-Host "Done. Activate with: `n.venv\\Scripts\\activate"

