#!/bin/sh
# Creates the Python environment the app uses for background training.
set -e
cd "$(dirname "$0")"
uv venv --system-site-packages --python 3.12 .venv
uv pip install --python .venv/bin/python wordfreq rapidfuzz numpy
.venv/bin/python -c "import torch" 2>/dev/null || uv pip install --python .venv/bin/python torch
echo "ok: training/.venv ready"
