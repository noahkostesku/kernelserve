from __future__ import annotations

import os
import sys
from pathlib import Path


def _sqlite3_available() -> bool:
    try:
        import sqlite3  # noqa: F401
        return True
    except ImportError:
        return False


def mlflow_sqlite_uri() -> tuple[str, bool]:
    """Return (tracking_uri, is_sqlite). Falls back to file:// if sqlite3 is unavailable."""
    if not _sqlite3_available():
        fallback = Path.home() / ".kernelserve" / "mlruns"
        fallback.mkdir(parents=True, exist_ok=True)
        return f"file://{fallback}", False
    db_path = os.environ.get("KERNELSERVE_MLFLOW_DB") or str(
        Path.home() / ".kernelserve" / "mlflow.db"
    )
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}", True
