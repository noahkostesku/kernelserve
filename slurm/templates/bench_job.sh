#!/bin/bash -l
# ==============================================================================
# bench_job.sh — Benchmark job (Narval A100 40 GB, non-interactive)
#
# Runs all three backends (cuda_oxide, triton, pytorch) for a given kernel,
# logs results to MLflow under kernelserve/<kernel>/<backend>/<cluster>/<YYYY-MM>.
#
# Submit via: /slurm-submit bench <kernel_name> [options]
# The /slurm-submit command expands ${PLACEHOLDER} directives below before
# calling sbatch; do not submit this file directly with sbatch.
# ==============================================================================
#
# ── Nibi / H100 overrides (Phase 2 only — do not activate until Phase 2) ──────
#   GPU_TYPE=h100
#   GPU_COUNT=<1-8>              # Nibi supports up to 8 per node
#   MEM_PER_GPU=80G
#   CUDA_ARCH=sm_90              # Phase 2 only; requires llvm/21+ (build manually)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --job-name=ks-bench-${KERNEL_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:${GPU_TYPE}:${GPU_COUNT}
#SBATCH --mem-per-gpu=${MEM_PER_GPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SCRATCH}/kernelserve-logs/%j/job.out
#SBATCH --error=${SCRATCH}/kernelserve-logs/%j/job.err

set -euo pipefail

# ── Configurable variables ────────────────────────────────────────────────────
CUDA_ARCH="${CUDA_ARCH:-sm_80}"
MEM_PER_GPU="${MEM_PER_GPU:-40G}"
GPU_TYPE="${GPU_TYPE:-a100}"
GPU_COUNT="${GPU_COUNT:-1}"
KERNEL_NAME="${KERNEL_NAME:-all}"
TIME_LIMIT="${TIME_LIMIT:-1:00:00}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/def-cbravo/$USER/kernelserve}"
LOG_DIR="${SCRATCH}/kernelserve-logs/${SLURM_JOB_ID}"

export CUDA_ARCH

# ── Teardown trap ─────────────────────────────────────────────────────────────
TRITON_PID=""
cleanup() {
    [[ -n "${TRITON_PID}" ]] && kill "${TRITON_PID}" 2>/dev/null || true
    [[ -n "${TRITON_PID}" ]] && wait "${TRITON_PID}" 2>/dev/null || true
    "${PROJECT_ROOT}/slurm/scripts/teardown.sh"
}
trap cleanup EXIT

# ── Job metadata ──────────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"
echo "======================================================"
echo "SLURM_JOB_ID : ${SLURM_JOB_ID}"
echo "Hostname     : $(hostname -f)"
echo "CUDA arch    : ${CUDA_ARCH}"
echo "Kernel       : ${KERNEL_NAME}"
echo "GPU type     : ${GPU_TYPE} × ${GPU_COUNT}"
echo "Log dir      : ${LOG_DIR}"
echo "======================================================"
nvidia-smi
echo "======================================================"

# ── Environment ───────────────────────────────────────────────────────────────
# Note: llvm/18.1.8 is the highest verified module on Narval; satisfies sm_80 ≥ 18.
module load StdEnv/2023 gcc/12.3 cuda/12.2 llvm/18.1.8 rust/1.91.0 python/3.11

export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MLFLOW_TRACKING_URI="file://${SCRATCH}/mlruns"

source "${PROJECT_ROOT}/slurm/scripts/setup_env.sh"

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
    echo "ERROR: .venv not found at ${PROJECT_ROOT}/.venv — run scripts/setup_env.sh first." >&2
    exit 1
fi
source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}"

# ── Build cuda-oxide kernels ──────────────────────────────────────────────────
echo "=== Building cuda-oxide kernels (arch=${CUDA_ARCH}) ==="
cd kernels/cuda_oxide
cargo oxide build --release
cd "${PROJECT_ROOT}"

# ── Start Triton Inference Server in background ───────────────────────────────
echo "=== Starting Triton Inference Server ==="
tritonserver \
    --model-repository="${PROJECT_ROOT}/serving/triton_backends" \
    --log-verbose=1 \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002 \
    > "${LOG_DIR}/triton.log" 2>&1 &
TRITON_PID=$!

# ── Wait for Triton to be ready (up to 180 s) ─────────────────────────────────
echo "Waiting for Triton to be ready..."
READY=0
for i in $(seq 1 36); do
    if curl -sf http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
        echo "Triton ready (attempt ${i})"
        READY=1
        break
    fi
    sleep 5
done
if [[ "${READY}" -eq 0 ]]; then
    echo "ERROR: Triton did not become ready within 180 s. See ${LOG_DIR}/triton.log" >&2
    exit 1
fi

# ── Run perf_analyzer for each backend ───────────────────────────────────────
MONTH="$(date +%Y-%m)"
for BACKEND in cuda_oxide triton pytorch; do
    echo "=== perf_analyzer: ${BACKEND} ==="
    perf_analyzer \
        -m "${KERNEL_NAME}_${BACKEND}" \
        -u localhost:8001 \
        --protocol grpc \
        --measurement-interval=5000 \
        --concurrency-range=1:4 \
        --output-shared-memory=system \
        > "${LOG_DIR}/perf_${BACKEND}.txt" 2>&1 \
        || echo "WARN: perf_analyzer failed for ${BACKEND} — check ${LOG_DIR}/perf_${BACKEND}.txt"

    # Log each backend to its own MLflow experiment (5-segment name required)
    srun --ntasks=1 python "${PROJECT_ROOT}/experiments/mlflow_setup.py" log \
        --experiment "kernelserve/${KERNEL_NAME}/${BACKEND}/${CC_CLUSTER:-narval}/${MONTH}" \
        --kernel "${KERNEL_NAME}" \
        --backend "${BACKEND}" \
        --perf-results "${LOG_DIR}/perf_${BACKEND}.txt" \
        --cluster "${CC_CLUSTER:-narval}"
done

echo "=== Benchmark job ${SLURM_JOB_ID} complete ==="
echo "Logs : ${LOG_DIR}"
seff "${SLURM_JOB_ID}" || true
