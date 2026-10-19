"""Reproduce the bid qualification review analysis using only the supplied CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path


FIELDS = (
    "audit_id",
    "submitted_at",
    "cert_synced_at",
    "division",
    "cert_type",
    "valid_days_left",
    "doc_page_count",
    "is_passed",
)
THRESHOLDS_MINUTES = (10, 15, 20, 25, 30, 40, 50)
FAILURE_LOSS_HOURS = 90
WARNING_COST_HOURS = 15


def load_data(path: Path) -> tuple[list[dict], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
        raw = list(reader)

    unique = []
    seen_rows = set()
    seen_ids = {}
    for row in raw:
        values = tuple(row[name] for name in FIELDS)
        if values in seen_rows:
            continue
        seen_rows.add(values)
        audit_id = row["audit_id"]
        if audit_id in seen_ids:
            raise ValueError(f"Conflicting records for audit_id={audit_id}")
        seen_ids[audit_id] = values

        submitted = datetime.fromisoformat(row["submitted_at"].replace("Z", "+00:00"))
        synced = datetime.fromisoformat(row["cert_synced_at"].replace("Z", "+00:00"))
        if submitted.tzinfo is None or synced.tzinfo is None:
            raise ValueError(f"Timestamp without time zone: audit_id={audit_id}")
        age_seconds = (submitted - synced).total_seconds()
        if age_seconds < 0:
            raise ValueError(f"Sync time after submission: audit_id={audit_id}")
        if row["is_passed"] not in ("", "0", "1"):
            raise ValueError(f"Unexpected outcome: audit_id={audit_id}")
        days_left = int(row["valid_days_left"])
        page_count = int(row["doc_page_count"])
        if days_left < 0 or page_count < 0:
            raise ValueError(f"Negative input: audit_id={audit_id}")

        unique.append(
            {
                "audit_id": audit_id,
                "date": submitted.date().isoformat(),
                "sync_age_seconds": age_seconds,
                "valid_days_left": days_left,
                "doc_page_count": page_count,
                "outcome": row["is_passed"],
            }
        )
    return raw, unique


def wilson_interval(successes: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.96
    p = successes / n
    denominator = 1 + z * z / n
    midpoint = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [round(midpoint - half_width, 4), round(midpoint + half_width, 4)]


def evaluate(rows: list[dict], threshold: int) -> dict:
    labeled = [r for r in rows if r["outcome"] in ("0", "1")]
    flagged = [r for r in labeled if r["sync_age_seconds"] >= threshold * 60]
    unflagged = [r for r in labeled if r["sync_age_seconds"] < threshold * 60]
    failures = sum(r["outcome"] == "0" for r in labeled)
    flagged_failures = sum(r["outcome"] == "0" for r in flagged)
    unflagged_failures = failures - flagged_failures
    warning_hours = WARNING_COST_HOURS * len(flagged)
    maximum_avoided_hours = FAILURE_LOSS_HOURS * flagged_failures
    required_prevented = warning_hours // FAILURE_LOSS_HOURS + 1 if flagged_failures else None
    return {
        "labeled": len(labeled),
        "failures": failures,
        "flagged": len(flagged),
        "flagged_failures": flagged_failures,
        "flagged_passes": len(flagged) - flagged_failures,
        "flagged_failure_rate": round(flagged_failures / len(flagged), 4) if flagged else None,
        "flagged_failure_rate_wilson_95pct": wilson_interval(flagged_failures, len(flagged)),
        "unflagged": len(unflagged),
        "unflagged_failures": unflagged_failures,
        "unflagged_failure_rate": round(unflagged_failures / len(unflagged), 4) if unflagged else None,
        "failure_capture_rate": round(flagged_failures / failures, 4) if failures else None,
        "warning_cost_hours": warning_hours,
        "maximum_avoidable_failure_hours": maximum_avoided_hours,
        "maximum_net_hours": maximum_avoided_hours - warning_hours,
        "break_even_prevention_share": round(warning_hours / maximum_avoided_hours, 4)
        if maximum_avoided_hours else None,
        "minimum_failures_to_prevent_for_positive_net": required_prevented,
        "pending_flagged": sum(
            r["outcome"] == "" and r["sync_age_seconds"] >= threshold * 60 for r in rows
        ),
    }


def validity_summary(rows: list[dict], maximum_days: int) -> dict:
    selected = [
        r for r in rows
        if r["outcome"] in ("0", "1") and r["valid_days_left"] <= maximum_days
    ]
    failures = sum(r["outcome"] == "0" for r in selected)
    return {
        "labeled": len(selected),
        "failures": failures,
        "failure_rate": round(failures / len(selected), 4) if selected else None,
        "maximum_net_hours": failures * FAILURE_LOSS_HOURS - len(selected) * WARNING_COST_HOURS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/bid_qualifications.csv"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    raw, unique = load_data(args.input)
    dates = sorted({r["date"] for r in unique})
    if len(dates) < 4:
        raise ValueError("Need at least four submission dates for a chronological holdout")
    holdout_dates = dates[-2:]
    train = [r for r in unique if r["date"] not in holdout_dates]
    holdout = [r for r in unique if r["date"] in holdout_dates]

    candidates = {str(t): evaluate(train, t) for t in THRESHOLDS_MINUTES}
    # Training-only choice; earlier warning wins an equal-score tie.
    selected = max(THRESHOLDS_MINUTES, key=lambda t: (candidates[str(t)]["maximum_net_hours"], -t))
    result = {
        "source": {
            "path": args.input.as_posix(),
            "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "raw_rows": len(raw),
            "exact_duplicate_rows": len(raw) - len(unique),
            "unique_audits": len(unique),
            "labeled_audits": sum(r["outcome"] in ("0", "1") for r in unique),
            "passed_audits": sum(r["outcome"] == "1" for r in unique),
            "failed_audits": sum(r["outcome"] == "0" for r in unique),
            "pending_audits": sum(r["outcome"] == "" for r in unique),
            "submission_dates": dates,
        },
        "assumptions": {
            "failure_loss_hours": FAILURE_LOSS_HOURS,
            "manual_warning_cost_hours": WARNING_COST_HOURS,
            "intervention_effectiveness": "unknown; maximum assumes every flagged failure is prevented",
        },
        "selection": {
            "train_dates": dates[:-2],
            "holdout_dates": holdout_dates,
            "candidate_thresholds_minutes": list(THRESHOLDS_MINUTES),
            "criterion": "maximum theoretical net hours on training data; lower threshold wins tie",
            "training_candidates": candidates,
            "selected_sync_age_threshold_minutes": selected,
        },
        "selected_rule": {
            "train": evaluate(train, selected),
            "holdout": evaluate(holdout, selected),
            "all_labeled": evaluate(unique, selected),
        },
        "validity_only_comparison": {
            "rule": "valid_days_left <= 3",
            "train": validity_summary(train, 3),
            "holdout": validity_summary(holdout, 3),
            "all_labeled": validity_summary(unique, 3),
            "zero_days_left_all_labeled": validity_summary(unique, 0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    h = result["selected_rule"]["holdout"]
    print(f"Unique audits: {len(unique)}; labeled: {result['source']['labeled_audits']}; pending: {result['source']['pending_audits']}")
    print(f"Selected age threshold: {selected} minutes")
    print(f"Holdout: {h['flagged_failures']}/{h['flagged']} flagged audits failed; {h['unflagged_failures']}/{h['unflagged']} unflagged audits failed")
    print(f"Holdout break-even: prevent at least {h['minimum_failures_to_prevent_for_positive_net']} of {h['flagged_failures']} flagged failures if each warning costs 15 hours")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
