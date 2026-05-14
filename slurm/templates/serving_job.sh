#!/bin/bash -l
# ==============================================================================
# serving_job.sh — Long-running serving job: Triton + Ray Serve + OTel
#                  (Narval A100 40 GB, up to 12 h)
#
# Keeps Triton Inference Server, Ray Serve, and the OTel collector alive until
# walltime or manual scancel.
#
# Submit via: /slurm-submit serve [options]
# The /slurm-submit command expands ${PLACEHOLDER} directives below before
# calling sbatch; do not submit this file directly with sbatch.
# ==============================================================================
#
# ── Nibi / H100 overrides (Phase 2 only — do not activate until Phase 2) ──────
#   GPU_TYPE=h100
#   GPU_COUNT=<1-8>
#   MEM_PER_GPU=80G
#   CUDA_ARCH=sm_90              # Phase 2 only; requires llvm/21+ (build manually)
#   TIME_LIMIT=<up to 24:00:00 on Nibi>
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --job-name=ks-serve
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
GPU_TYPE="${GPU_TYPE:-a100}"
GPU_COUNT="${GPU_COUNT:-1}"
MEM_PER_GPU="${MEM_PER_GPU:-40G}"
TIME_LIMIT="${TIME_LIMIT:-8:00:00}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/def-cbravo/$USER/kernelserve}"
LOG_DIR="${SCRATCH}/kernelserve-logs/${SLURM_JOB_ID}"

export CUDA_ARCH

# ── Teardown trap ─────────────────────────────────────────────────────────────
TRITON_PID=""
SERVE_PID=""
OTEL_PID=""
cleanup() {
    echo "=== Shutting down serving stack ==="
    [[ -n "${SERVE_PID}" ]]  && kill "${SERVE_PID}"  2>/dev/null || true
    [[ -n "${OTEL_PID}" ]]   && kill "${OTEL_PID}"   2>/dev/null || true
    ray stop --force 2>/dev/null || true
    [[ -n "${TRITON_PID}" ]] && kill "${TRITON_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
    "${PROJECT_ROOT}/slurm/scripts/teardown.sh"
}
trap cleanup EXIT

# ── Job metadata ──────────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"
echo "======================================================"
echo "SLURM_JOB_ID : ${SLURM_JOB_ID}"
echo "Hostname     : $(hostname -f)"
echo "CUDA arch    : ${CUDA_ARCH}"
echo "GPU type     : ${GPU_TYPE} × ${GPU_COUNT}"
echo "Time limit   : ${TIME_LIMIT}"
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
export TRITON_GRPC_URL="localhost:8001"
export TRITON_HTTP_URL="localhost:8000"

source "${PROJECT_ROOT}/slurm/scripts/setup_env.sh"

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
    echo "ERROR: .venv not found at ${PROJECT_ROOT}/.venv — run scripts/setup_env.sh first." >&2
    exit 1
fi
source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}"

# ── Start Ray local cluster ───────────────────────────────────────────────────
# Local mode only — do NOT use ray.init(address="auto"); it hangs on HPC.
echo "=== Starting Ray (local head) ==="
ray start \
    --head \
    --num-gpus="${GPU_COUNT}" \
    --num-cpus=4 \
    --disable-usage-stats \
    > "${LOG_DIR}/ray.log" 2>&1
echo "Ray head started. Dashboard: http://$(hostname -f):8265"

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

# ── Deploy Ray Serve application ──────────────────────────────────────────────
echo "=== Deploying Ray Serve app ==="
srun --ntasks=1 serve run \
    "${PROJECT_ROOT}/serving/ray_serve/deployment.py" \
    > "${LOG_DIR}/serve.log" 2>&1 &
SERVE_PID=$!

# ── Start OpenTelemetry collector ─────────────────────────────────────────────
echo "=== Starting OTel collector ==="
"${PROJECT_ROOT}/observability/otel-collector/start.sh" \
    > "${LOG_DIR}/otel.log" 2>&1 &
OTEL_PID=$!

# ── Print live endpoint information ──────────────────────────────────────────
NODE_HOST="$(hostname -f)"
cat <<EOF

=== KernelServe Serving Stack is Live ===
Job ID       : ${SLURM_JOB_ID}
Triton HTTP  : http://${NODE_HOST}:8000
Triton gRPC  : ${NODE_HOST}:8001
Triton metrics: http://${NODE_HOST}:8002/metrics
Ray dashboard: http://${NODE_HOST}:8265
Log dir      : ${LOG_DIR}

SSH tunnel from laptop:
  ssh -L 8000:${NODE_HOST}:8000 -L 8265:${NODE_HOST}:8265 <username>@narval.alliancecan.ca
=========================================

EOF

# ── Hold until walltime (SLURM will send SIGTERM at end) ─────────────────────
echo "Sleeping until walltime — scancel ${SLURM_JOB_ID} to terminate early."
sleep infinity
