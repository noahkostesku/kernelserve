#!/bin/bash
#SBATCH --account=def-cbravo
#SBATCH --job-name=kernelserve-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --time=0-01:00
#SBATCH --output=$SCRATCH/kernelserve-logs/%j-out.txt
#SBATCH --error=$SCRATCH/kernelserve-logs/%j-err.txt

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0 python/3.11

export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2/lib64:${LD_LIBRARY_PATH:-}
export MLFLOW_TRACKING_URI="file://$SCRATCH/mlruns"

REPO_DIR="$HOME/projects/def-cbravo/$USER/kernelserve"
VENV_DIR="$SCRATCH/kernelserve-venv"

# ── Activate Python venv ─────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR. Run scripts/setup_env.sh first." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"

# ── Build Rust kernels ────────────────────────────────────────────────────────
echo "=== Building cuda-oxide kernels ==="
cd kernels/cuda_oxide
cargo build --release
cd "$REPO_DIR"

# ── Run benchmark suite ───────────────────────────────────────────────────────
echo "=== Running GPU benchmark suite ==="
RESULTS_FILE="$SCRATCH/kernelserve-logs/${SLURM_JOB_ID}-bench.json"

pytest tests/benchmark/ -m gpu \
    --benchmark-json="$RESULTS_FILE" \
    --tb=short \
    -v

# ── Log to MLflow ─────────────────────────────────────────────────────────────
echo "=== Logging results to MLflow ==="
python experiments/mlflow_setup.py log \
    --kernel "${KERNEL_NAME:-all}" \
    --results "$RESULTS_FILE" \
    --commit "$(git rev-parse --short HEAD)" \
    --cluster narval

echo "=== Benchmark job ${SLURM_JOB_ID} complete ==="
echo "Results: $RESULTS_FILE"
echo "Run: seff ${SLURM_JOB_ID}"
