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

# ── PyO3 bindings ─────────────────────────────────────────────────────────────
echo "=== Building PyO3 CPU bindings into .venv ==="
pushd kernels/pyo3_bindings >/dev/null
# Use the uv .venv's maturin and Python so the extension is importable via uv run.
env -u CONDA_PREFIX VIRTUAL_ENV="$REPO_DIR/.venv" "$REPO_DIR/.venv/bin/maturin" develop \
    2>&1 || echo "WARNING: maturin develop failed. Re-run 'make build-bindings' after fixing."
popd >/dev/null

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Dev setup complete ==="
echo "Run CPU-only tests:   uv run pytest -m 'not gpu'  (or: make test)"
echo "Run GPU tests:        uv run pytest -m gpu  (requires A100)"
echo "Rebuild bindings:     make build-bindings"
echo "Check deps:           bash scripts/check_deps.sh"
