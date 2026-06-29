from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

STEP_RE = re.compile(r"^\s*\d+\s*\.\s*(VALID|INVALID)\s*$", re.IGNORECASE | re.MULTILINE)

def _as_bool(x: Any):
    return x if x is True or x is False else None

def _gold(r: dict):
    for key in ("actual_correctness", "generator_correct", "correct"): 
        if key in r and _as_bool(r[key]) is not None:
            return _as_bool(r[key])
    return None

def _difficulty(r: dict) -> str: 
    b = r.get("budget", {}) or {}
    return b.get("difficulty") or r.get("difficulty") or "unknown"


def step_counts(text: str | None) -> tuple[int, int]:
    """Return (total_step_judgments, n_invalid) parsed from verifier raw output."""
    if not text:
        return 0, 0
    judgments = [m.group(1).upper() for m in STEP_RE.finditer(text)]
    total = len(judgments)
    n_invalid = sum(1 for j in judgments if j == "INVALID")
    return total, n_invalid


def extract_ratio(filename: str) -> float | None:
    m = re.search(r"_vr(\d+)", filename) or re.search(r"_r(\d+)", filename)
    return round(int(m.group(1)) / 100, 4) if m else None


def row_flags(r: dict) -> dict[str, Any]:
    total_steps, n_invalid = step_counts(r.get("verifier_raw_output"))
    all_steps_valid = total_steps > 0 and n_invalid == 0
    answer_missing = bool(r.get("truncated")) or not r.get("predicted_answer")
    mismatch = bool(all_steps_valid and answer_missing)
    return {
        "id": r.get("id"), 
        "difficulty": _difficulty(r), 
        "total_steps": total_steps, 
        "n_invalid_steps": n_invalid, 
        "all_steps_valid": all_steps_valid, 
        "answer_missing": answer_missing, 
        "gold": _gold(r), 
        "verifier_decision": r.get("verifier_decision"), 
        "process_outcome_mismatch": mismatch,
    }


def _rate(counts: dict) -> dict[str, Any]:
    n = counts["n_valid"]
    return {
        "n_valid": n,
        "mismatch_count": counts["mismatch"],
        "answer_missing_count": counts["answer_missing"],
        "all_steps_valid_count": counts["all_steps_valid"],
        "process_outcome_mismatch_rate": counts["mismatch"] / n if n else None,
        "answer_missing_rate": counts["answer_missing"] / n if n else None,     
    }


def process_folder(input_dir: str, glob: str, output_dir: str | None) -> None:
    in_dir = Path(input_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Not a directory: {in_dir}")
    out_dir = Path(output_dir) if output_dir else in_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.rglob(glob)
                   if "summary" not in p.name and "analysis" not in str(p)
                   and not p.name.endswith("_process_mismatch.jsonl"))
    if not files:
        raise SystemExit(f"No files matching {glob!r} under {in_dir}")

    overall = defaultdict(int)
    by_ratio: dict[str, Any] = {}
    by_difficulty = defaultdict(lambda: defaultdict(int))

    print(f"Found {len(files)} file(s).\n")
    for p in files:
        ratio = extract_ratio(p.name)
        rkey = f"ratio={ratio:.2f}" if ratio is not None else p.stem
        rc = defaultdict(int)
        sidecar = out_dir / f"{p.stem}_process_mismatch.jsonl"
        with open(p, "r", encoding="utf-8") as f, open(sidecar, "w", encoding="utf-8") as out:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    continue
                flags = row_flags(r)
                out.write(json.dumps(flags, ensure_ascii=False) + "\n")

                rc["n_valid"] += 1
                overall["n_valid"] += 1
                d = flags["difficulty"]
                by_difficulty[d]["n_valid"] += 1
                if flags["all_steps_valid"]:
                    rc["all_steps_valid"] += 1; overall["all_steps_valid"] += 1
                    by_difficulty[d]["all_steps_valid"] += 1
                if flags["answer_missing"]:
                    rc["answer_missing"] += 1; overall["answer_missing"] += 1
                    by_difficulty[d]["answer_missing"] += 1
                if flags["process_outcome_mismatch"]:
                    rc["mismatch"] += 1; overall["mismatch"] += 1
                    by_difficulty[d]["mismatch"] += 1 

        stats = _rate(rc)
        by_ratio[rkey] = {"file": p.name, "verifier_ratio": ratio, **stats}
        rate = stats["process_outcome_mismatch_rate"]
        rate_s = f"{rate:.2%}" if rate is not None else "n/a"
        print(f"  {p.name:<40} mismatch {rate_s:>7}  ({rc['mismatch']}/{rc['n_valid']})  -> {sidecar.name}")

    aggregate = {
        "overall": _rate(overall),
        "by_ratio": by_ratio,
        "by_difficulty": {d: _rate(c) for d, c in sorted(by_difficulty.items())},
        "definition": {
            "process_outcome_mismatch": "all_steps_valid AND answer_missing",
            "all_steps_valid": "total_steps > 0 AND n_invalid == 0 (re-parsed from verifier_raw_output)",
            "answer_missing": "generator truncated OR predicted_answer empty",
        },
    }
    agg_path = out_dir / "process_outcome_mismatch.json"
    agg_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    ov = aggregate["overall"]
    ov_rate = ov["process_outcome_mismatch_rate"]
    print(f"\nOVERALL mismatch rate: "
          f"{(f'{ov_rate:.2%}' if ov_rate is not None else 'n/a')} "
          f"({ov['mismatch_count']}/{ov['n_valid']})")
    print(f"Aggregate written to {agg_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute process_outcome_mismatch for Experiment 2 from existing JSONLs (no re-run)."
    )
    ap.add_argument("--input-dir", required=True, help="Directory containing gsm8k_vr*.jsonl files")
    ap.add_argument("--glob", default="*_vr*.jsonl", help="Glob for result files (default: *_vr*.jsonl)")
    ap.add_argument("--output-dir", default=None, help="Output dir (default: <input-dir>/analysis)")
    args = ap.parse_args()
    process_folder(args.input_dir, args.glob, args.output_dir)


if __name__ == "__main__":
    main()
