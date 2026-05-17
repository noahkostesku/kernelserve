from __future__ import annotations

import sys
from pathlib import Path

_CLUSTERS: dict[str, dict] = {
    "narval": {
        "cuda_arch": "sm_80",
        "gpu_type": "a100",
        "mem_directive": "#SBATCH --mem=32G",
        "module_lines": [
            "module load StdEnv/2023 gcc/12.3 cuda/12.9 llvm/18.1.8 clang/18.1.8 rust/1.91.0 python/3.11",
        ],
        "arch_path": "x86-64-v3",
        "llvm_version": "21.1.5",
        "clang_version": "18.1.8",
        "cuda_version": "12.9.1",
    },
    "nibi": {
        "cuda_arch": "sm_90",
        "gpu_type": "h100",
        "mem_directive": "#SBATCH --mem-per-gpu=80G",
        "module_lines": [
            "module load StdEnv/2023 gcc/12.3 llvm/21.1.5 clang/18.1.8 rust/1.91.0 python/3.11",
            "module load gcc/14.3 cuda/12.9",
        ],
        "arch_path": "x86-64-v4",
        "llvm_version": "21.1.5",
        "clang_version": "18.1.8",
        "cuda_version": "12.9.1",
    },
}


def _find_project_root() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / "kernels").is_dir():
        print(
            f"error: cannot locate project root from {__file__!r}; "
            f"expected 'kernels/' directory at {root}",
            file=sys.stderr,
        )
        sys.exit(1)
    return root


def _render(kernel: str, cluster: str, account: str, cfg: dict) -> str:
    arch_path = cfg["arch_path"]
    llvm_ver = cfg["llvm_version"]
    clang_ver = cfg["clang_version"]
    cuda_ver = cfg["cuda_version"]
    gpu_type = cfg["gpu_type"]
    cuda_arch = cfg["cuda_arch"]

    parts: list[str] = [
        "#!/bin/bash",
        f"#SBATCH --job-name=ks-bench-{kernel}-{cluster}",
        f"#SBATCH --account={account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=8",
        f"#SBATCH --gpus-per-node={gpu_type}:1",
        cfg["mem_directive"],
        "#SBATCH --time=0-00:45:00",
        "#SBATCH --output=/scratch/%u/kernelserve-logs/%j-out.txt",
        "#SBATCH --error=/scratch/%u/kernelserve-logs/%j-err.txt",
        "",
        "set -euo pipefail",
        "",
        "# ── Environment ──────────────────────────────────────────────────────────────",
    ]
    parts += cfg["module_lines"]
    if cluster == "nibi":
        parts += [
            f"export CUDA_ARCH={cuda_arch}",
            f"export CLUSTER={cluster}",
        ]
    parts += [
        "export PATH=$HOME/.cargo/bin:$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/bin:$PATH",
        "export CUDA_OXIDE_BACKEND=$SCRATCH/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so",
        'export RUSTFLAGS="-L $HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib"',
        (
            "export LD_LIBRARY_PATH=$HOME/.rustup/toolchains/nightly-2026-04-03-x86_64-unknown-linux-gnu/lib"
            f":/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/{cuda_ver}/lib64"
            ":${LD_LIBRARY_PATH:-}"
        ),
        f"export LIBCLANG_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/{arch_path}/Compiler/gcccore/clang/{clang_ver}/lib",
        f"export CUDA_OXIDE_LLC=/cvmfs/soft.computecanada.ca/easybuild/software/2023/{arch_path}/Compiler/gcccore/llvmcore/{llvm_ver}/bin/llc",
        'export MLFLOW_TRACKING_URI="file://$SCRATCH/mlruns"',
        "",
        'REPO_DIR="$SCRATCH/kernelserve"',
        "",
        "# ── Activate Python venv ─────────────────────────────────────────────────────",
        'if [[ ! -d "$REPO_DIR/.venv" ]]; then',
        '    echo "ERROR: venv not found at $REPO_DIR/.venv. Run: python -m venv .venv && uv sync" >&2',
        "    exit 1",
        "fi",
        "# shellcheck source=/dev/null",
        'source "$REPO_DIR/.venv/bin/activate"',
        "",
        'cd "$REPO_DIR"',
        "",
        "# ── Correctness check (exit early if kernel is wrong) ─────────────────────────",
        f'echo "=== {kernel} correctness check ==="',
        f'cd "$REPO_DIR/kernels/{kernel}_standalone"',
        'cargo oxide run --arch "${CUDA_ARCH:-sm_80}" --verbose',
        'cd "$REPO_DIR"',
        "",
        "# ── Build release binary ──────────────────────────────────────────────────────",
        f'echo "=== Building {kernel} release binary ==="',
        f'cd "$REPO_DIR/kernels/{kernel}_standalone"',
        'cargo oxide build --arch "${CUDA_ARCH:-sm_80}"',
        'cd "$REPO_DIR"',
        "",
        f'export CUDA_OXIDE_BIN="$REPO_DIR/kernels/{kernel}_standalone/target/release/{kernel}"',
        "",
        "# ── OTel trace export ─────────────────────────────────────────────────────────",
        'export OTEL_SERVICE_NAME="kernelserve-bench"',
        'mkdir -p "$SCRATCH/traces"',
        'export OTEL_SPAN_FILE="$SCRATCH/traces/bench_${SLURM_JOB_ID}.jsonl"',
        "",
        f'# ── Run benchmark ─────────────────────────────────────────────────────────────',
        f'echo "=== {kernel} benchmark ({cluster}) ==="',
        f'srun --export=ALL python "$REPO_DIR/experiments/bench_{kernel}.py"',
        "",
        'echo "=== Job ${SLURM_JOB_ID} complete ==="',
        'echo "Run: seff ${SLURM_JOB_ID}"',
        'echo "MLflow: ${MLFLOW_TRACKING_URI}"',
        'echo "Traces: ${OTEL_SPAN_FILE}"',
    ]

    return "\n".join(parts) + "\n"


def run_submit(args: object) -> None:
    cluster: str = args.cluster  # type: ignore[attr-defined]
    kernel: str = args.kernel  # type: ignore[attr-defined]
    account: str = args.account  # type: ignore[attr-defined]

    cfg = _CLUSTERS[cluster]
    script = _render(kernel, cluster, account, cfg)

    root = _find_project_root()
    out_dir = root / "slurm" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{kernel}_{cluster}.sh"
    out_path.write_text(script, encoding="utf-8")
    out_path.chmod(out_path.stat().st_mode | 0o111)

    print(f"Generated: {out_path}")
    print()
    print("To submit:")
    print(f"  sbatch {out_path}")
    print()
    print("After submitting:")
    print("  squeue -u $USER")
    print(f"  ks results --cluster {cluster}")
