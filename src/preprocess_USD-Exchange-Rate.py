"""Compatibility command for the team's original JISDOR cleaner filename.

The validated output now has schema date,jisdor in data/interim/jisdor/.
All CLI options are forwarded to the shared cleaner.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.clean_jisdor import main


if __name__ == "__main__":
    raise SystemExit(main())
