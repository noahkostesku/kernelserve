#!/usr/bin/env bash
# Stop all KernelServe processes: Ray, Triton server, and clean Ray session dirs.
# Run from the repo root or via `bash scripts/teardown.sh`

set -uo pipefail

echo "=== KernelServe teardown ==="

# ── Ray ───────────────────────────────────────────────────────────────────────
if command -v ray &>/dev/null; then
    echo "Stopping Ray cluster..."
    ray stop --force 2>/dev/null || true
else
    echo "ray not found, skipping"
fi

# ── Triton Inference Server ───────────────────────────────────────────────────
if pgrep -f tritonserver &>/dev/null; then
    echo "Killing tritonserver process(es)..."
    pkill -f tritonserver || true
else
    echo "No tritonserver process found"
fi

# ── Ray session dirs ──────────────────────────────────────────────────────────
echo "Cleaning Ray session dirs in /tmp..."
rm -rf /tmp/ray 2>/dev/null || true
find /tmp -maxdepth 1 -name 'ray_session_*' -exec rm -rf {} + 2>/dev/null || true

# ── Reminder ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Done ==="
echo "If you have live SLURM jobs, cancel them with:"
echo "  squeue -u \$USER"
echo "  scancel <job_id>"
