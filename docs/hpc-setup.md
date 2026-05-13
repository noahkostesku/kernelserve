# HPC Setup — Alliance Canada (Narval)

## Connecting

```bash
ssh <username>@narval.alliancecan.ca
```

Your username is your Alliance Canada account (CCDB). Multi-factor authentication is required.

## Directory layout on Narval

| Path | Quota | Use for |
|---|---|---|
| `$HOME` (~60 GB) | Backed up | Code, configs, small files |
| `$SCRATCH` (~20 TB, not backed up) | Large temp storage | Model files, MLflow runs, benchmark outputs |
| `$PROJECT` | Backed up | Shared project data (`def-cbravo`) |

Clone the repo to `$HOME/projects/def-cbravo/$USER/kernelserve` (or your preferred path).

## Module loads for Phase 1 (A100 sm_80)

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0 python/3.11
```

Add to your `~/.bashrc` or to the top of every SLURM script.

## Setting up the Python venv

```bash
# Run once after cloning
export SCRATCH_VENV="$SCRATCH/kernelserve-venv"
python3 -m venv "$SCRATCH_VENV"
source "$SCRATCH_VENV/bin/activate"
pip install --upgrade pip uv
uv sync
```

Then update `VENV_DIR` in `slurm/bench_job.sh` and `slurm/serving_job.sh` to match.

## Interactive GPU session (for debugging)

```bash
salloc --account=def-cbravo --gpus-per-node=a100:1 --mem=32G --time=1:00:00 --ntasks=1
# Once allocated:
module load StdEnv/2023 gcc/12.3 cuda/12.2 rust/1.91.0 python/3.11
source $SCRATCH/kernelserve-venv/bin/activate
nvidia-smi  # verify A100 is visible
```

## Submitting jobs

```bash
# Benchmark job (~1h)
sbatch slurm/bench_job.sh

# Serving job (~8h)
sbatch slurm/serving_job.sh

# Check queue
squeue -u $USER

# Check efficiency after job completes
seff <job_id>
```

## MLflow on Narval

```bash
export MLFLOW_TRACKING_URI="file://$SCRATCH/mlruns"
mlflow ui --port 5000 &

# From your laptop, SSH tunnel to view the UI:
ssh -L 5000:localhost:5000 <username>@narval.alliancecan.ca
# then open http://localhost:5000
```

## Useful commands

```bash
# Check GPU memory on an allocated node
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Check LLVM version available
module spider llvm

# Cancel a job
scancel <job_id>

# Storage quota
diskusage_report
```
