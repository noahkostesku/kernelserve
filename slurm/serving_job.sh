#!/bin/bash
#SBATCH --account=def-cbravo
#SBATCH --job-name=kernelserve-serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH --time=0-08:00
#SBATCH --output=$SCRATCH/kernelserve-logs/%j-out.txt
#SBATCH --error=$SCRATCH/kernelserve-logs/%j-err.txt

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11

export MLFLOW_TRACKING_URI="file://$SCRATCH/mlruns"
export TRITON_GRPC_URL="localhost:8001"
export TRITON_MODEL_NAME="cuda_oxide_backend"

REPO_DIR="$HOME/projects/def-cbravo/$USER/kernelserve"
VENV_DIR="$SCRATCH/kernelserve-venv"
LOG_DIR="$SCRATCH/kernelserve-logs"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

# ── Start Triton Inference Server ─────────────────────────────────────────────
echo "=== Starting Triton Inference Server ==="
# TODO: replace <tag> with the pinned nvcr.io/nvidia/tritonserver image tag
# See docs/serving-guide.md for the correct tag
tritonserver \
    --model-repository="$REPO_DIR/serving/triton_backends" \
    --log-verbose=1 \
    --metrics-port=8002 \
    > "$LOG_DIR/${SLURM_JOB_ID}-triton.log" 2>&1 &
TRITON_PID=$!

# Wait for Triton to be ready (up to 120s)
echo "Waiting for Triton to be ready..."
for i in $(seq 1 24); do
    if curl -sf http://localhost:8000/v2/health/ready; then
        echo "Triton is ready"
        break
    fi
    sleep 5
done

# ── Start Ray Serve ───────────────────────────────────────────────────────────
echo "=== Starting Ray Serve ==="
ray start --head --num-gpus=0 --num-cpus=4 \
    > "$LOG_DIR/${SLURM_JOB_ID}-ray.log" 2>&1

python serving/ray_serve/deployment.py \
    >> "$LOG_DIR/${SLURM_JOB_ID}-ray.log" 2>&1 &
RAY_PID=$!

echo "=== Serving job ${SLURM_JOB_ID} running ==="
echo "Triton PID: $TRITON_PID | Ray Serve PID: $RAY_PID"
echo "Logs: $LOG_DIR/${SLURM_JOB_ID}-*.log"

# Keep the job alive until time limit or manual scancel
wait $TRITON_PID
