"""Compatibility wrapper for :mod:`src.pipeline.run_task1`."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.run_task1 import main, run_task1


if __name__ == "__main__":
    raise SystemExit(main())

