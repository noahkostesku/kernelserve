VENV    := .venv
MATURIN := $(VENV)/bin/maturin
PYTHON  := $(VENV)/bin/python
PYTEST  := $(VENV)/bin/pytest

.PHONY: dev build-bindings build-bindings-gpu test lint

# Full dev bootstrap: install Python deps then build the PyO3 CPU binding.
dev:
	uv sync --dev
	$(MAKE) build-bindings

# Build PyO3 CPU binding into the uv .venv (no CUDA required).
# Must be re-run after any `uv sync` because uv removes packages it doesn't track.
build-bindings: $(MATURIN)
	cd kernels/pyo3_bindings && \
	  env -u CONDA_PREFIX VIRTUAL_ENV="$(CURDIR)/$(VENV)" "$(CURDIR)/$(MATURIN)" develop

# Build PyO3 GPU binding (requires cargo oxide build first to emit PTX).
build-bindings-gpu: $(MATURIN)
	cd kernels/cuda_oxide && cargo oxide build --release --features gpu
	cd kernels/pyo3_bindings && \
	  env -u CONDA_PREFIX VIRTUAL_ENV="$(CURDIR)/$(VENV)" "$(CURDIR)/$(MATURIN)" develop --features gpu

# Sync → rebuild bindings → run tests.
# Uses .venv/bin/pytest directly to avoid a second uv sync that would undo the maturin install.
test:
	uv sync --dev
	$(MAKE) build-bindings
	$(PYTEST) -m "not gpu"

lint:
	uv run ruff check .
	uv run mypy kernels serving observability experiments kernelserve

# Ensure the venv (and maturin inside it) exists before any build step.
$(MATURIN):
	uv sync --dev
