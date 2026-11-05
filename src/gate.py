#!/usr/bin/env python3
"""Command-line release gate: the same checks as Lab 3, for use in a CI build.

It computes the code-based evaluator scores from the captured traces, reads the calibrated judge
verdicts (data/judge_verdicts/revised_rubric.jsonl), compares a candidate agent version with a baseline, and
exits with code 1 if any check fails, so a build step can stop the release.

    python -m src.gate v2-nova
    python -m src.gate v2-prompt --baseline v1
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import utils

# Floors for behavior a claims agent must always get right, and the largest drop allowed against
# the baseline on the judge criterion that passed calibration in Lab 3.
FLOORS = {"decision": 1.0, "payout": 1.0, "no_payout_on_denial": 1.0, "pass^3": 0.95}
REGRESSION_CHECKS = {"next_step": 0.10}


def version_metrics() -> pd.DataFrame:
    code_scores = utils.code_scores()
    metrics = code_scores.groupby("version")[["decision", "payout", "no_payout_on_denial"]].mean()
    metrics["pass^3"] = code_scores.groupby(["version", "case_id"])["passed"].all().groupby("version").mean()
    judge_rates, _ = utils.pass_rates(utils.load_jsonl(utils.REVISED_RESULTS), utils.RUBRIC_REVISED)
    metrics["next_step"] = judge_rates["next_step"]
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidate", help="agent version to check, for example v2-nova")
    ap.add_argument("--baseline", default="v1", help="agent version currently in production")
    args = ap.parse_args()

    metrics = version_metrics()
    if args.candidate not in metrics.index or args.baseline not in metrics.index:
        print(f"Unknown version. Available: {', '.join(metrics.index)}")
        return 2
    was, got = metrics.loc[args.baseline], metrics.loc[args.candidate]

    failed = []
    for name, floor in FLOORS.items():
        ok = got[name] >= floor - 1e-9
        print(f"  {name:<22} {got[name]:.3f}   floor {floor:.2f}   {'pass' if ok else 'FAIL'}")
        failed += [] if ok else [name]
    for name, tolerance in REGRESSION_CHECKS.items():
        drop = was[name] - got[name]
        ok = drop <= tolerance + 1e-9
        print(f"  {name:<22} {got[name]:.3f}   baseline {was[name]:.3f}, largest allowed drop {tolerance:.2f}   "
              f"{'pass' if ok else 'FAIL'}")
        failed += [] if ok else [name]

    if failed:
        print(f"\nGATE FAIL for {args.candidate}: {', '.join(failed)}")
        return 1
    print(f"\nGATE PASS for {args.candidate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
