"""Structured logging for pipeline runs."""

import logging
import sys
import time
from pathlib import Path


def setup_logging(out_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger; optionally mirror to ``out_dir/run.log``."""
    logger = logging.getLogger("deeppipe")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "run.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log(msg: str) -> None:
    """Timestamped log line (backward-compatible helper)."""
    logging.getLogger("deeppipe").info(msg)


def log_legacy(msg: str) -> None:
    """Print-style log used during migration."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
