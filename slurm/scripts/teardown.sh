#!/bin/bash -l
# Called by the EXIT trap in every SLURM job script.
# Executed as a subprocess, so venv deactivation is a no-op (the subprocess
# has no parent shell function to call). Triton and Ray are killed by PID
# signal or pkill so they don't outlive the allocation.

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-unknown}"

echo "=== teardown: job ${JOB_ID} ==="

# ── Stop Triton Inference Server ──────────────────────────────────────────────
if pgrep -f tritonserver > /dev/null 2>&1; then
    echo "teardown: stopping tritonserver"
    pkill -SIGTERM -f tritonserver || true
    # give it up to 10 s to flush before SIGKILL
    for i in $(seq 1 10); do
        pgrep -f tritonserver > /dev/null 2>&1 || break
        sleep 1
    done
    pkill -SIGKILL -f tritonserver 2>/dev/null || true
else
    echo "teardown: tritonserver not running"
fi

# ── Stop Ray ──────────────────────────────────────────────────────────────────
if command -v ray > /dev/null 2>&1 && ray status > /dev/null 2>&1; then
    echo "teardown: stopping Ray"
    ray stop --force 2>/dev/null || true
else
    echo "teardown: Ray not running"
fi

# ── Deactivate Python venv ────────────────────────────────────────────────────
# `deactivate` is a shell function injected by venv activation; it is not
# available in this subprocess. The line below is a no-op here but is correct
# when teardown.sh is sourced directly in a dev session.
deactivate 2>/dev/null || true

echo "=== teardown complete: job ${JOB_ID} ==="
