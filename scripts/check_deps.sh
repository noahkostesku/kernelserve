#!/usr/bin/env bash
# Verify that all KernelServe dependencies are present and at the right versions.
# Prints PASS/WARN/FAIL for each check.

set -uo pipefail

PASS="\033[32mPASS\033[0m"
WARN="\033[33mWARN\033[0m"
FAIL="\033[31mFAIL\033[0m"
failures=0

check() {
    local label="$1"; shift
    if "$@" &>/dev/null; then
        echo -e "[$PASS] $label"
    else
        echo -e "[$FAIL] $label"
        failures=$((failures + 1))
    fi
}

warn() {
    local label="$1"
    echo -e "[$WARN] $label"
}

# ── Python ────────────────────────────────────────────────────────────────────
check "Python >= 3.11" python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"

# ── Triton ────────────────────────────────────────────────────────────────────
check "triton importable" python3 -c "import triton"

# ── PyTorch ───────────────────────────────────────────────────────────────────
check "torch importable" python3 -c "import torch"

# ── CUDA availability ─────────────────────────────────────────────────────────
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" &>/dev/null; then
    echo -e "[$PASS] CUDA available (torch.cuda.is_available() = True)"
else
    warn "CUDA not available — GPU tests will be skipped (expected on macOS/no-GPU hosts)"
fi

if command -v nvidia-smi &>/dev/null; then
    check "nvidia-smi" nvidia-smi --query-gpu=name --format=csv,noheader
else
    warn "nvidia-smi not found — expected on macOS / CPU-only hosts"
fi

# ── LLVM version ──────────────────────────────────────────────────────────────
# sm_80 (A100, Phase 1) works with LLVM 18+
# sm_90 (H100, Phase 2) requires LLVM 21+
if command -v llvm-config &>/dev/null; then
    LLVM_VER=$(llvm-config --version | cut -d. -f1)
    if [[ "$LLVM_VER" -ge 21 ]]; then
        echo -e "[$PASS] LLVM $LLVM_VER (supports sm_80 and sm_90)"
    elif [[ "$LLVM_VER" -ge 18 ]]; then
        echo -e "[$PASS] LLVM $LLVM_VER (supports sm_80 / A100 Phase 1)"
        warn "LLVM $LLVM_VER: sm_90 / H100 Phase 2 requires LLVM >= 21"
    else
        echo -e "[$FAIL] LLVM $LLVM_VER: sm_80 requires LLVM >= 18"
        failures=$((failures + 1))
    fi
else
    warn "llvm-config not found in PATH (may be loaded via module on Narval)"
fi

# ── Rust / cargo ──────────────────────────────────────────────────────────────
check "cargo present" cargo --version

# ── MLflow ────────────────────────────────────────────────────────────────────
check "mlflow importable" python3 -c "import mlflow"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
if [[ "$failures" -eq 0 ]]; then
    echo -e "All checks passed."
else
    echo -e "\033[31m$failures check(s) failed.\033[0m"
    exit 1
fi
