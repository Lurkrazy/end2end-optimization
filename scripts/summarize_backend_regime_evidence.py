#!/usr/bin/env python3
"""Summarize backend-vs-regime harness outputs without using an LLM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_BASELINE = "sglang-triton-bf16-baseline"


def load_summaries(root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/summary.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_summary_path"] = str(path)
        summaries.append(data)
    if not summaries:
        raise SystemExit(f"No */summary.json files found under {root}")
    return summaries


def metric_value(regime: dict[str, Any], metric: str) -> float | None:
    value = regime.get(metric, {}).get("mean")
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_speedup(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def build_report(
    summaries: list[dict[str, Any]],
    metric: str,
    baseline_id: str,
) -> dict[str, Any]:
    regime_names = sorted(
        {
            regime_name
            for summary in summaries
            for regime_name in summary.get("regimes", {}).keys()
        }
    )

    by_submission = {
        summary.get("submission_id", Path(summary["_summary_path"]).parent.name): summary
        for summary in summaries
    }
    baseline = by_submission.get(baseline_id)

    rows: list[dict[str, Any]] = []
    winners: dict[str, str] = {}

    for regime_name in regime_names:
        best_submission: str | None = None
        best_value: float | None = None

        baseline_value = None
        if baseline:
            baseline_regime = baseline.get("regimes", {}).get(regime_name, {})
            baseline_value = metric_value(baseline_regime, metric)

        for summary in summaries:
            submission_id = summary.get("submission_id", Path(summary["_summary_path"]).parent.name)
            regime = summary.get("regimes", {}).get(regime_name)
            if not regime:
                continue
            value = metric_value(regime, metric)
            reliable = bool(regime.get("reliable", False))
            ok = bool(summary.get("ok", False))
            speedup = value / baseline_value if value is not None and baseline_value else None

            if ok and reliable and value is not None and (best_value is None or value > best_value):
                best_value = value
                best_submission = submission_id

            rows.append(
                {
                    "regime": regime_name,
                    "submission_id": submission_id,
                    "ok": ok,
                    "reliable": reliable,
                    metric: value,
                    "speedup_vs_baseline": speedup,
                    "summary_path": summary["_summary_path"],
                }
            )

        if best_submission:
            winners[regime_name] = best_submission

    return {
        "metric": metric,
        "baseline_id": baseline_id,
        "baseline_found": baseline is not None,
        "regime_count": len(regime_names),
        "submission_count": len(summaries),
        "winners": winners,
        "winner_inversion": len(set(winners.values())) > 1,
        "rows": rows,
        "failed_summaries": [
            {
                "submission_id": summary.get("submission_id", Path(summary["_summary_path"]).parent.name),
                "summary_path": summary["_summary_path"],
                "error": summary.get("error"),
            }
            for summary in summaries
            if not summary.get("ok", False)
        ],
    }


def print_markdown(report: dict[str, Any]) -> None:
    metric = report["metric"]
    print(f"# Backend/regime reproduction summary\n")
    print(f"- Metric: `{metric}.mean`")
    print(f"- Baseline: `{report['baseline_id']}` ({'found' if report['baseline_found'] else 'missing'})")
    print(f"- Winner inversion: `{report['winner_inversion']}`")
    print()

    print("## Per-regime winners")
    print()
    print("| regime | winner |")
    print("|---|---|")
    for regime, winner in sorted(report["winners"].items()):
        print(f"| {regime} | {winner} |")
    print()

    print("## All cells")
    print()
    print("| regime | submission | ok | reliable | metric | speedup vs baseline |")
    print("|---|---|---:|---:|---:|---:|")
    for row in sorted(report["rows"], key=lambda r: (r["regime"], r["submission_id"])):
        print(
            "| {regime} | {submission_id} | {ok} | {reliable} | {metric_value} | {speedup} |".format(
                regime=row["regime"],
                submission_id=row["submission_id"],
                ok=str(row["ok"]).lower(),
                reliable=str(row["reliable"]).lower(),
                metric_value=fmt_float(row[metric]),
                speedup=fmt_speedup(row["speedup_vs_baseline"]),
            )
        )

    if report["failed_summaries"]:
        print()
        print("## Failed summaries")
        print()
        print("| submission | summary | error |")
        print("|---|---|---|")
        for failed in report["failed_summaries"]:
            error = failed.get("error") or {}
            message = str(error.get("message", "")).replace("|", "\\|")
            print(f"| {failed['submission_id']} | {failed['summary_path']} | {message} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Directory containing one subdir per spec")
    parser.add_argument(
        "--metric",
        choices=("req_per_s", "tokens_per_s", "wall_s"),
        default="req_per_s",
        help="Metric to compare. Default: req_per_s",
    )
    parser.add_argument(
        "--baseline-id",
        default=DEFAULT_BASELINE,
        help=f"Submission id used as speedup baseline. Default: {DEFAULT_BASELINE}",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of Markdown")
    args = parser.parse_args()

    report = build_report(load_summaries(args.result_dir), args.metric, args.baseline_id)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_markdown(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
