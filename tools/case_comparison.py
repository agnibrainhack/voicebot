"""Rebuild case_comparison.html (expected vs actual, case by case) for an existing campaign folder.

    python tools/case_comparison.py --campaign output/campaign_out
    python tools/case_comparison.py --campaign output/campaign_llm_sut

Reads dataset/case_results.jsonl, dataset/turn_by_turn.csv, dataset/turn_by_turn_summary.csv and report.json
from the folder; nothing is re-run. A campaign written before case_results.jsonl existed still works from the
turn-by-turn files (verdict and turns), without the per-stage checks and failure reasons.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmsjourney.case_compare import write_case_comparison  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", default="output/campaign_out", help="campaign output folder")
    ap.add_argument("--out", default="case_comparison.html", help="file name inside the folder")
    a = ap.parse_args()
    if not (Path(a.campaign) / "dataset").exists():
        sys.exit(f"{a.campaign}: no dataset/ folder - is this a campaign output folder?")
    print(write_case_comparison(a.campaign, a.out))


if __name__ == "__main__":
    main()
