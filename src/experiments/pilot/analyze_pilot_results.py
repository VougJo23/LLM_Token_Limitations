from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

from src.utils.io import load_jsonl


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(mean(values)) if values else 0.0


def _as_bool(x: Any) -> bool:
    return bool(x is True)


@dataclass(frozen=True)
class Metrics:
    n_total: int
    n_valid: int
    generator_accuracy: float
    verifier_accuracy: float
    lazy_accept_rate: float
    false_reject_rate: float
    abstain_rate: float
    generator_truncation_rate: float
    verifier_truncation_rate: float
    avg_generator_total_tokens: float
    avg_verifier_total_tokens: float


def compute_metrics(rows: list[dict[str, Any]]) -> Metrics:
    valid = [r for r in rows if "error" not in r]

    return Metrics(
        n_total=len(rows),
        n_valid=len(valid),
        generator_accuracy=_safe_mean(_as_bool(r.get("generator_correct") or r.get("actual_correctness")) for r in valid),
        verifier_accuracy=_safe_mean(_as_bool(r.get("verifier_correct")) for r in valid),
        lazy_accept_rate=_safe_mean(_as_bool(r.get("verifier_lazy_accept")) for r in valid),
        false_reject_rate=_safe_mean(_as_bool(r.get("verifier_false_reject")) for r in valid),
        abstain_rate=_safe_mean(_as_bool(r.get("verifier_abstained")) for r in valid),
        generator_truncation_rate=_safe_mean(_as_bool(r.get("truncated")) for r in valid),
        verifier_truncation_rate=_safe_mean(_as_bool(r.get("verifier_truncated")) for r in valid),
        avg_generator_total_tokens=_safe_mean((r.get("total_tokens_used") or r.get("usage", {}).get("generator", {}).get("total_tokens") or 0) for r in valid),
        avg_verifier_total_tokens=_safe_mean((r.get("verifier_total_tokens_used") or r.get("usage", {}).get("verifier", {}).get("total_tokens") or 0) for r in valid),
    )


def group_and_summarize(
    rows: list[dict[str, Any]],
    group_by: str,
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        if group_by.startswith("budget."):
            key = r.get("budget", {}).get(group_by.removeprefix("budget."))
        else:
            key = r.get(group_by)
        buckets[str(key)].append(r)

    summary: dict[str, Any] = {
        "group_by": group_by,
        "groups": {},
    }

    for key, group_rows in sorted(buckets.items(), key=lambda kv: kv[0]):
        m = compute_metrics(group_rows)
        summary["groups"][key] = {
            **m.__dict__,
        }

    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize pilot JSONL results (lazy-accept, abstain, truncation, accuracy).")
    p.add_argument("--input", required=True, help="Path to a pilot .jsonl file")
    p.add_argument(
        "--group-by",
        default="budget.generator_ratio",
        help="Field to group by (e.g. budget.generator_ratio, verifier_max_tokens, config)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional path to write summary JSON (defaults to stdout only)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rows = list(load_jsonl(args.input))

    overall = compute_metrics(rows)
    grouped = group_and_summarize(rows, group_by=args.group_by)

    out = {
        "input": args.input,
        "overall": overall.__dict__,
        "grouped": grouped,
    }

    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
