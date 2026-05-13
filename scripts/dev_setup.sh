#!/usr/bin/env bash
# Local development environment bootstrap.
# Run once after cloning: bash scripts/dev_setup.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ── Python ────────────────────────────────────────────────────────────────────
echo "=== Checking Python version ==="
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED="3.11"
if ! python3 -c "import sys; assert sys.version_info >= (3,11), 'need 3.11+'" 2>/dev/null; then
    echo "ERROR: Python >= $REQUIRED required, found $PYTHON_VERSION" >&2
    exit 1
fi
echo "Python $PYTHON_VERSION OK"

echo "=== Installing uv ==="
if ! command -v uv &>/dev/null; then
    pip install --quiet uv
fi
uv --version

echo "=== Installing Python dependencies ==="
uv sync --dev
echo "Python deps installed in .venv/"

# ── Rust ──────────────────────────────────────────────────────────────────────
echo "=== Checking Rust toolchain ==="
if ! command -v rustup &>/dev/null; then
    echo "ERROR: rustup not found. Install from https://rustup.rs" >&2
    exit 1
fi
rustup show

echo "=== Building cuda-oxide kernels (may fail without CUDA headers) ==="
pushd kernels/cuda_oxide >/dev/null
cargo build 2>&1 || echo "WARNING: cargo build failed — expected on macOS without CUDA. Skipping."
popd >/dev/null

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Dev setup complete ==="
echo "Run CPU-only tests:   uv run pytest -m 'not gpu'"
echo "Run GPU tests:        uv run pytest -m gpu  (requires A100)"
echo "Check deps:           bash scripts/check_deps.sh"
