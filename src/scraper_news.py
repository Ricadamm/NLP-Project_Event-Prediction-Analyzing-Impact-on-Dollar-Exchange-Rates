"""Compatibility CLI for the Task 1 GDELT candidate collector.

Example: python src/scraper_news.py --pilot
Explicit dates (or --pilot) are required to prevent accidental full scraping.
"""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.acquisition.collect_gdelt import main

if __name__ == "__main__":
    sys.exit(main())
