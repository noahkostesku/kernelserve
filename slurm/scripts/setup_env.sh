#!/bin/bash -l
# Sourced by SLURM job scripts to establish a consistent environment.
# Usage: source "${PROJECT_ROOT}/slurm/scripts/setup_env.sh"
#
# Requires PROJECT_ROOT to be set (or defaults to Alliance Canada convention).
# Safe to source multiple times; module operations are idempotent.

set -euo pipefail

# ── Project root ──────────────────────────────────────────────────────────────
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/def-cbravo/$USER/kernelserve}"

# ── Modules ───────────────────────────────────────────────────────────────────
module purge

# TODO: verify exact module names on Narval with `module spider <name>`
module load StdEnv/2023                  # required base; pins compiler ABI
module load gcc/12.3                     # TODO: verify on Narval
module load cuda/12.2                    # TODO: verify exact patch version (12.2.x)
module load llvm/18.1.8                  # TODO: llvm/21 not yet available as a module on Narval;
                                         #       18.1.8 satisfies sm_80 (≥ 18) requirement.
                                         #       Build llvm/21 manually under $SCRATCH if needed for sm_90.
module load rust/1.91.0                  # TODO: rust/nightly not available; 1.91.0 is latest on Narval.
                                         #       If nightly features are required, install via rustup post-load.
module load python/3.11                  # TODO: verify on Narval

# ── CUDA paths ────────────────────────────────────────────────────────────────
# TODO: confirm this cvmfs prefix matches the cuda/12.2 module on your allocation
export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"

# ── GPU architecture ──────────────────────────────────────────────────────────
# Read from env; never hardcode sm_80 or sm_90 in source files.
export CUDA_ARCH="${CUDA_ARCH:-sm_80}"

# ── Cluster identity ──────────────────────────────────────────────────────────
# CC_CLUSTER is set by the Alliance Canada module system on compute nodes.
# Provide a fallback so local/dry-run sourcing doesn't error.
export CC_CLUSTER="${CC_CLUSTER:-narval}"

# ── Python venv ───────────────────────────────────────────────────────────────
if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
    echo "ERROR: .venv not found at ${PROJECT_ROOT}/.venv" >&2
    echo "       Create it with: python3 -m venv .venv && source .venv/bin/activate && uv sync" >&2
    return 1 2>/dev/null || exit 1
fi
source "${PROJECT_ROOT}/.venv/bin/activate"

# ── MLflow ────────────────────────────────────────────────────────────────────
export MLFLOW_TRACKING_URI="file://${SCRATCH}/mlruns"

echo "setup_env: CC_CLUSTER=${CC_CLUSTER}  CUDA_ARCH=${CUDA_ARCH}  venv=${PROJECT_ROOT}/.venv"
