#!/bin/bash
# Keep interpreters and dependencies inside the repository.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_PYTHON_BIN_DIR="$PWD/.python-bin"
if [ ! -x .bootstrap/bin/uv ]; then
    python3 -m venv .bootstrap
    .bootstrap/bin/python -m pip install 'uv==0.13.6'
fi
.bootstrap/bin/uv python install 3.10.18
if [ ! -x .venv/bin/python ]; then
    .bootstrap/bin/uv venv --python 3.10.18 .venv
fi
if [ ! -x .venv-original/bin/python ]; then
    .bootstrap/bin/uv venv --python 3.10.18 .venv-original
fi
models_requirements=reproduction/requirements-models.txt
original_requirements=reproduction/requirements-original.txt
if [ -f reproduction/environment-models-freeze.txt ]; then
    models_requirements=reproduction/environment-models-freeze.txt
fi
if [ -f reproduction/environment-original-freeze.txt ]; then
    original_requirements=reproduction/environment-original-freeze.txt
fi
.bootstrap/bin/python -m pip --python .venv/bin/python install --timeout 180 -r "$models_requirements"
.bootstrap/bin/python -m pip --python .venv-original/bin/python install --timeout 180 -r "$original_requirements"
echo 'Environments ready. On macOS, LightGBM also needs libomp; see reproduction/README_复现说明.md.'
