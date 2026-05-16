#!/bin/bash
#SBATCH --job-name=kernelserve-rms-norm-phase4-nibi
#SBATCH --account=def-cbravo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=0-00:45:00
#SBATCH --output=/scratch/%u/kernelserve-logs/%j-out.txt
#SBATCH --error=/scratch/%u/kernelserve-logs/%j-err.txt

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
# TODO(nibi): verify cuda module version on Nibi — Narval uses cuda/12.9; Nibi likely 12.6 or 12.9
# TODO(nibi): verify llvm module name on Nibi — module may be llvmcore/21.1.5 or llvm/21.x; must be ≥ 21 for sm_90
# TODO(nibi): verify clang module name/version on Nibi — Narval uses clang/18.1.8
module load StdEnv/2023 gcc/12.3 cuda/12.6 llvmcore/21.1.5 clang/18.1.8 rust/1.91.0 python/3.11

# sm_90 targets H100 on Nibi; read by cargo oxide build and bench_rms_norm.py
export CUDA_ARCH=sm_90
# MLflow cluster tag read by experiments/bench_rms_norm.py for experiment name segment
export CLUSTER=nibi

export PATH=$HOME/.cargo/bin:$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/bin:$PATH
export CUDA_OXIDE_BACKEND=$SCRATCH/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
export RUSTFLAGS="-L $HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib"
# TODO(nibi): update cudacore cvmfs path to match actual cuda version loaded above
export LD_LIBRARY_PATH=$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/bin:$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib:/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.0/lib64:${LD_LIBRARY_PATH:-}
# TODO(nibi): verify LIBCLANG_PATH — clang/18.1.8 cvmfs path may differ on Nibi
export LIBCLANG_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/clang/18.1.8/lib
# TODO(nibi): confirm llc binary is reachable at this cvmfs path from Nibi nodes
export CUDA_OXIDE_LLC=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/llvmcore/21.1.5/bin/llc
export MLFLOW_TRACKING_URI="file://$SCRATCH/mlruns"

REPO_DIR="$SCRATCH/kernelserve"

# ── Activate Python venv ─────────────────────────────────────────────────────
if [[ ! -d "$REPO_DIR/.venv" ]]; then
    echo "ERROR: venv not found at $REPO_DIR/.venv. Run: python -m venv .venv && uv sync" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR"

# ── Correctness check (exit early if kernel is wrong) ─────────────────────────
echo "=== rms-norm correctness check (${CUDA_ARCH}) ==="
cd "$REPO_DIR/kernels/rms_norm_standalone"
cargo oxide run --arch "${CUDA_ARCH}" --verbose
cd "$REPO_DIR"

# ── Build release binary (used by bench_rms_norm.py via subprocess) ───────────
echo "=== Building rms-norm release binary (${CUDA_ARCH}) ==="
cd "$REPO_DIR/kernels/rms_norm_standalone"
cargo oxide build --arch "${CUDA_ARCH}"
cd "$REPO_DIR"

export CUDA_OXIDE_BIN="$REPO_DIR/kernels/rms_norm_standalone/target/release/rms_norm"

# ── OTel trace export (file-based; no sidecar needed) ─────────────────────────
export OTEL_SERVICE_NAME="kernelserve-bench"
mkdir -p "$SCRATCH/traces"
export OTEL_SPAN_FILE="$SCRATCH/traces/bench_${SLURM_JOB_ID}.jsonl"
# OTEL_EXPORTER_OTLP_ENDPOINT is intentionally unset → ConsoleSpanExporter activates

# ── Run Phase 4 benchmark: 3 backends × 3 shapes → MLflow + OTel spans ───────
echo "=== Phase 4 benchmark on Nibi H100 (${CUDA_ARCH}, 3 backends × 3 shapes) ==="
srun python "$REPO_DIR/experiments/bench_rms_norm.py"

echo "=== Job ${SLURM_JOB_ID} complete ==="
echo "Run: seff ${SLURM_JOB_ID}"
echo "MLflow: ${MLFLOW_TRACKING_URI}"
echo "Traces: ${OTEL_SPAN_FILE}"
