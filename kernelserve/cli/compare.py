from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import argparse

_COL = 16

_GPU_DEFAULTS = dict(warmup=20, iters=200, batch=2048, hidden_dim=4096)
_CPU_DEFAULTS = dict(warmup=2,  iters=5,   batch=128,  hidden_dim=512)


def _is_cpu_mode(args: argparse.Namespace) -> bool:
    if getattr(args, "fast", False):
        return True
    if os.environ.get("KERNELSERVE_DEVICE", "").lower() == "cpu":
        return True
    return not torch.cuda.is_available()


def _time_fn(fn: object, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()  # type: ignore[operator]
    times_us = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()  # type: ignore[operator]
        times_us.append((time.perf_counter() - t0) * 1e6)
    times_us.sort()
    return times_us[iters // 2], times_us[int(iters * 0.99)]


def run_compare(args: argparse.Namespace) -> None:
    import kernelserve

    cpu_mode = _is_cpu_mode(args)
    defaults = _CPU_DEFAULTS if cpu_mode else _GPU_DEFAULTS

    warmup     = defaults["warmup"]
    iters      = defaults["iters"]
    batch      = args.batch      if args.batch      is not None else defaults["batch"]
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else defaults["hidden_dim"]

    if cpu_mode:
        print(
            "WARNING: Running in CPU mock mode — results are not representative "
            "of GPU performance.",
            file=sys.stderr,
        )

    x = torch.randn(batch, hidden_dim, dtype=torch.float32)
    w = torch.ones(hidden_dim, dtype=torch.float32)
    bytes_accessed = (2 * batch * hidden_dim + hidden_dim) * 4

    ref = torch.nn.functional.rms_norm(x, (hidden_dim,), weight=w)
    rows: list[tuple[str, float, float, float, float]] = []

    # cuda-oxide / cpu-ref path
    backend_name = "cuda_oxide" if (
        os.environ.get("KERNELSERVE_DEVICE", "").lower() != "cpu"
        and torch.cuda.is_available()
    ) else "cpu_ref"
    p50, p99 = _time_fn(lambda: kernelserve.rms_norm(x, w), warmup, iters)
    out_ks = kernelserve.rms_norm(x, w)
    max_err = float((out_ks - ref).abs().max())
    rows.append((backend_name, p50, p99, bytes_accessed / (p50 / 1e6) / 1e9, max_err))

    # PyTorch reference
    p50, p99 = _time_fn(
        lambda: torch.nn.functional.rms_norm(x, (hidden_dim,), weight=w), warmup, iters
    )
    rows.append(("pytorch", p50, p99, bytes_accessed / (p50 / 1e6) / 1e9, 0.0))

    # Triton (Linux + CUDA only)
    try:
        sys.path.insert(0, "kernels/triton")
        from rms_norm import rms_norm as triton_rms_norm  # type: ignore[import]
        p50, p99 = _time_fn(lambda: triton_rms_norm(x, w, eps=1e-5), warmup, iters)
        out_tri = triton_rms_norm(x, w, eps=1e-5)
        max_err_tri = float((out_tri - ref).abs().max())
        rows.append(("triton", p50, p99, bytes_accessed / (p50 / 1e6) / 1e9, max_err_tri))
    except ImportError:
        rows.append(("triton", float("nan"), float("nan"), float("nan"), float("nan")))

    header = (
        f"{'backend':<{_COL}} {'p50_us':>10} {'p99_us':>10} {'GB/s':>10} {'max_abs_err':>14}"
    )
    sep = "-" * len(header)
    print(f"\nkernel={args.kernel}  batch={batch}  hidden_dim={hidden_dim}")
    print(sep)
    print(header)
    print(sep)
    for name, p50, p99, gbs, err in rows:
        print(
            f"{name:<{_COL}} {p50:>10.2f} {p99:>10.2f} {gbs:>10.2f} {err:>14.2e}"
        )
    print()
