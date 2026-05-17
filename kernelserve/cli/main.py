from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="ks", description="KernelServe CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    bench_p = sub.add_parser("bench", help="Benchmark a kernel and print p50/p99 metrics")
    bench_p.add_argument("--kernel", default="rms_norm", choices=["rms_norm"])
    bench_p.add_argument("--batch", type=int, default=None)
    bench_p.add_argument("--hidden-dim", type=int, default=None)
    bench_p.add_argument("--warmup", type=int, default=None)
    bench_p.add_argument("--iters", type=int, default=None)
    bench_p.add_argument("--log-mlflow", action="store_true", help="Log results to MLflow")
    bench_p.add_argument(
        "--fast",
        action="store_true",
        help="Force reduced config (warmup=3, iters=10, 128×512) for quick smoke tests",
    )

    compare_p = sub.add_parser("compare", help="Side-by-side comparison of all backends")
    compare_p.add_argument("--kernel", default="rms_norm", choices=["rms_norm"])
    compare_p.add_argument("--batch", type=int, default=None)
    compare_p.add_argument("--hidden-dim", type=int, default=None)
    compare_p.add_argument(
        "--fast",
        action="store_true",
        help="Force reduced config (warmup=2, iters=5, 128×512) for quick smoke tests",
    )

    new_kernel_p = sub.add_parser(
        "new-kernel",
        help="Scaffold a new kernel with stub files",
    )
    new_kernel_p.add_argument(
        "name",
        help="Kernel name (valid Python/Rust identifier, e.g. softmax)",
    )

    submit_p = sub.add_parser(
        "submit",
        help="Generate a SLURM job script for a kernel and print the sbatch command",
    )
    submit_p.add_argument("--kernel", required=True, choices=["rms_norm"])
    submit_p.add_argument("--cluster", required=True, choices=["narval", "nibi"])
    submit_p.add_argument("--account", required=True, help="SLURM account (e.g. def-cbravo)")

    args = parser.parse_args()

    if args.command == "bench":
        from kernelserve.cli.bench import run_bench
        run_bench(args)
    elif args.command == "compare":
        from kernelserve.cli.compare import run_compare
        run_compare(args)
    elif args.command == "new-kernel":
        from kernelserve.cli.new_kernel import run_new_kernel
        run_new_kernel(args)
    elif args.command == "submit":
        from kernelserve.cli.submit import run_submit
        run_submit(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
