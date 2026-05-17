from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import patch

from kernelserve.cli.submit import _render, run_submit


def _make_args(
    kernel: str = "rms_norm", cluster: str = "narval", account: str = "def-cbravo"
) -> object:
    ns = types.SimpleNamespace(kernel=kernel, cluster=cluster, account=account)
    return ns


# ── _render unit tests ────────────────────────────────────────────────────────

def test_narval_script_has_sm80():
    script = _render("rms_norm", "narval", "def-cbravo", _narval_cfg())
    assert "sm_80" in script
    assert "a100" in script
    assert "x86-64-v3" in script
    assert script.count("module load") == 1


def test_narval_has_mem_directive():
    script = _render("rms_norm", "narval", "def-cbravo", _narval_cfg())
    assert "#SBATCH --mem=32G" in script
    assert "--mem-per-gpu" not in script


def test_narval_no_explicit_arch_export():
    script = _render("rms_norm", "narval", "def-cbravo", _narval_cfg())
    assert "export CUDA_ARCH" not in script
    assert "export CLUSTER" not in script


def test_nibi_script_has_sm90():
    script = _render("rms_norm", "nibi", "def-cbravo", _nibi_cfg())
    assert "sm_90" in script
    assert "h100" in script
    assert "x86-64-v4" in script
    assert script.count("module load") == 2


def test_nibi_has_mem_per_gpu_directive():
    script = _render("rms_norm", "nibi", "def-cbravo", _nibi_cfg())
    assert "#SBATCH --mem-per-gpu=80G" in script
    assert "#SBATCH --mem=32G" not in script


def test_nibi_exports_cuda_arch_and_cluster():
    script = _render("rms_norm", "nibi", "def-cbravo", _nibi_cfg())
    assert "export CUDA_ARCH=sm_90" in script
    assert "export CLUSTER=nibi" in script


def test_account_in_sbatch_directive():
    script = _render("rms_norm", "narval", "def-special", _narval_cfg())
    assert "#SBATCH --account=def-special" in script


def test_kernel_name_in_script():
    script = _render("rms_norm", "narval", "def-cbravo", _narval_cfg())
    assert "rms_norm" in script
    assert "bench_rms_norm.py" in script


def test_script_starts_with_shebang():
    script = _render("rms_norm", "narval", "def-cbravo", _narval_cfg())
    assert script.startswith("#!/bin/bash\n")


def test_script_ends_with_newline():
    script = _render("rms_norm", "narval", "def-cbravo", _narval_cfg())
    assert script.endswith("\n")


# ── run_submit integration tests ──────────────────────────────────────────────

def test_output_written_to_generated_dir(tmp_path: Path):
    with patch("kernelserve.cli.submit._find_project_root", return_value=tmp_path):
        (tmp_path / "kernels").mkdir()
        run_submit(_make_args(kernel="rms_norm", cluster="narval", account="def-cbravo"))
    out = tmp_path / "slurm" / "generated" / "rms_norm_narval.sh"
    assert out.exists()
    assert out.stat().st_size > 0


def test_output_file_is_executable(tmp_path: Path):
    with patch("kernelserve.cli.submit._find_project_root", return_value=tmp_path):
        (tmp_path / "kernels").mkdir()
        run_submit(_make_args(kernel="rms_norm", cluster="narval", account="def-cbravo"))
    out = tmp_path / "slurm" / "generated" / "rms_norm_narval.sh"
    assert out.stat().st_mode & 0o111


def test_stdout_contains_sbatch_command(tmp_path: Path, capsys):
    with patch("kernelserve.cli.submit._find_project_root", return_value=tmp_path):
        (tmp_path / "kernels").mkdir()
        run_submit(_make_args(kernel="rms_norm", cluster="narval", account="def-cbravo"))
    captured = capsys.readouterr()
    assert "sbatch" in captured.out
    assert "rms_norm_narval.sh" in captured.out


def test_stdout_contains_post_submit_hints(tmp_path: Path, capsys):
    with patch("kernelserve.cli.submit._find_project_root", return_value=tmp_path):
        (tmp_path / "kernels").mkdir()
        run_submit(_make_args(kernel="rms_norm", cluster="nibi", account="def-cbravo"))
    captured = capsys.readouterr()
    assert "squeue -u $USER" in captured.out
    assert "--cluster nibi" in captured.out


def test_nibi_output_path(tmp_path: Path):
    with patch("kernelserve.cli.submit._find_project_root", return_value=tmp_path):
        (tmp_path / "kernels").mkdir()
        run_submit(_make_args(kernel="rms_norm", cluster="nibi", account="def-cbravo"))
    out = tmp_path / "slurm" / "generated" / "rms_norm_nibi.sh"
    assert out.exists()


# ── helpers ───────────────────────────────────────────────────────────────────

def _narval_cfg() -> dict:
    from kernelserve.cli.submit import _CLUSTERS
    return _CLUSTERS["narval"]


def _nibi_cfg() -> dict:
    from kernelserve.cli.submit import _CLUSTERS
    return _CLUSTERS["nibi"]
