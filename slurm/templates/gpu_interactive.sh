#!/bin/bash -l
# ==============================================================================
# gpu_interactive.sh — Interactive GPU session launcher (Narval A100 40 GB)
#
# Usage:
#   bash slurm/templates/gpu_interactive.sh
#   GPU_COUNT=2 TIME_LIMIT=4:00:00 bash slurm/templates/gpu_interactive.sh
#
# This script calls salloc and drops you into a login shell on the compute node.
# Submit via: /slurm-submit interactive [options]
# ==============================================================================
#
# ── Nibi / H100 overrides (Phase 2 only — do not activate until Phase 2) ──────
#   GPU_TYPE=h100
#   GPU_COUNT=<1-8>
#   MEM_PER_GPU=80G
#   CUDA_ARCH=sm_90            # Phase 2 only; requires llvm/21+ (build manually)
#   Override example:
#     GPU_TYPE=h100 MEM_PER_GPU=80G CUDA_ARCH=sm_90 bash gpu_interactive.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configurable variables ────────────────────────────────────────────────────
SLURM_ACCOUNT="${SLURM_ACCOUNT:-def-cbravo}"
GPU_TYPE="${GPU_TYPE:-a100}"
GPU_COUNT="${GPU_COUNT:-1}"
MEM_PER_GPU="${MEM_PER_GPU:-40G}"
TIME_LIMIT="${TIME_LIMIT:-2:00:00}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/def-cbravo/$USER/kernelserve}"

export CUDA_ARCH

# ── Teardown trap ─────────────────────────────────────────────────────────────
trap '"${PROJECT_ROOT}/slurm/scripts/teardown.sh"' EXIT

# ── Pre-flight info ───────────────────────────────────────────────────────────
cat <<EOF
=== KernelServe Interactive Session ===
Account    : ${SLURM_ACCOUNT}
GPU        : ${GPU_TYPE} × ${GPU_COUNT}   mem-per-gpu=${MEM_PER_GPU}
Time limit : ${TIME_LIMIT}
CUDA arch  : ${CUDA_ARCH}
Project    : ${PROJECT_ROOT}
========================================
Requesting allocation — once inside the node run:

  module load StdEnv/2023 gcc/12.3 cuda/12.2 llvm/18.1.8 rust/1.91.0 python/3.11
  # Note: llvm/18.1.8 is the highest verified version on Narval (satisfies sm_80 ≥ 18).
  # llvm/21 is not available as a module; build manually under \$SCRATCH if needed.
  export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2
  export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
  source \${PROJECT_ROOT}/.venv/bin/activate
  nvidia-smi

EOF

# ── Launch allocation ─────────────────────────────────────────────────────────
salloc \
    --account="${SLURM_ACCOUNT}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gres="gpu:${GPU_TYPE}:${GPU_COUNT}" \
    --mem-per-gpu="${MEM_PER_GPU}" \
    --time="${TIME_LIMIT}" \
    srun --export=ALL,CUDA_ARCH="${CUDA_ARCH}",PROJECT_ROOT="${PROJECT_ROOT}" \
         --pty bash -l
