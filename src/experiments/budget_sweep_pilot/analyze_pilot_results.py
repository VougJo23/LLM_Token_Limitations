import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean

from src.utils.io import load_jsonl


def _as_bool(x):
    if x is True:
        return True
    if x is False:
        return False
    return None


def _actual_correctness(r):
    # Prefer canonical key used by the pilot pipeline.
    for key in ("actual_correctness", "generator_correct", "correct"):
        if key in r:
            v = _as_bool(r.get(key))
            if v is not None:
                return v
    return None


def compute_metrics(rows):
    valid = [r for r in rows if "error" not in r]
    decided = [r for r in valid if r.get("verifier_decision") is not None]

    total_valid = len(valid)
    total_decided = len(decided)

    correct_accepts = 0
    false_rejects = 0
    correct_rejections = 0
    lazy_accepts = 0

    for r in decided:
        gold = _actual_correctness(r)
        pred = _as_bool(r.get("verifier_decision"))
        if gold is None or pred is None:
            continue

        if gold is True and pred is True:
            correct_accepts += 1
        elif gold is True and pred is False:
            false_rejects += 1
        elif gold is False and pred is False:
            correct_rejections += 1
        elif gold is False and pred is True:
            lazy_accepts += 1

    fpr_denom = lazy_accepts + correct_rejections
    fnr_denom = false_rejects + correct_accepts

    false_positive_rate = lazy_accepts / fpr_denom if fpr_denom else 0.0
    false_negative_rate = false_rejects / fnr_denom if fnr_denom else 0.0
    error_detection_rate = correct_rejections / fpr_denom if fpr_denom else 0.0

    system_level_accuracy = (
        (correct_accepts + correct_rejections) / total_decided
        if total_decided else 0.0
    )

    effective_acceptance_rate = total_decided / total_valid if total_valid else 0.0

    def avg_bool(key):
        return mean([bool(r.get(key)) for r in valid]) if valid else 0.0

    gen_tokens = [r.get("completion_tokens") or 0 for r in valid]
    ver_tokens = [r.get("verifier_completion_tokens") or 0 for r in valid]
    total_tokens_used = sum(gen_tokens) + sum(ver_tokens)
    n_correct = sum(1 for r in valid if _actual_correctness(r) is True)
    tokens_per_correct = total_tokens_used / n_correct if n_correct else 0.0

    generator_ratio = None
    for r in valid:
        gr = r.get("budget", {}).get("generator_ratio")
        if gr is not None:
            generator_ratio = float(gr)
            break

    return {
        "data_quality": {
            "n_total": len(rows),
            "n_valid": total_valid,
            "n_decided": total_decided,
        },

        "system_performance": {
            "system_accuracy_after_verification": system_level_accuracy,
            "generator_accuracy": avg_bool("generator_correct"),
        },

        "verification": {
            "error_detection_rate": error_detection_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "accept_rate": (correct_accepts + lazy_accepts) / total_decided if total_decided else 0.0,
            "reject_rate": (correct_rejections + false_rejects) / total_decided if total_decided else 0.0,
            "abstain_rate": avg_bool("verifier_abstained"),
            "coverage": effective_acceptance_rate,
            "acceptance_breakdown": {
                "lazy_accept_rate": avg_bool("verifier_lazy_accept"),
                "correct_accepts": int(correct_accepts),
                "false_rejects": int(false_rejects),
                "correct_rejections": int(correct_rejections),
                "lazy_accepts": int(lazy_accepts),
            },
        },

        "efficiency": {
            "tokens_per_correct": float(tokens_per_correct),
            "generator_tokens_per_sample": mean(gen_tokens),
            "verifier_tokens_per_sample": mean(ver_tokens),
        },

        "budget": {
            "generator_ratio": generator_ratio,
        },
    }
    
def group_and_summarize(rows, group_by):
    buckets = defaultdict(list)

    for r in rows:
        if group_by.startswith("budget."):
            key = r.get("budget", {}).get(group_by.replace("budget.", ""))
        else:
            key = r.get(group_by)

        buckets[str(key)].append(r)

    grouped = {}

    for k, group in sorted(buckets.items()):
        grouped[k] = compute_metrics(group)

    return grouped


def _iter_jsonl_files(input_dir: Path, *, recursive: bool) -> list[Path]:
    globber = input_dir.rglob if recursive else input_dir.glob
    paths: list[Path] = []
    for p in sorted(globber("*.jsonl")):
        if p.name.endswith("_summary.jsonl"):
            continue
        if p.name in {"summary_points.jsonl", "false_positives.jsonl", "false_negatives.jsonl"}:
            continue
        paths.append(p)
    return paths


def _looks_like_pilot_output(rows: list[dict]) -> bool:
    for r in rows[:200]:
        if isinstance(r, dict) and "verifier_decision" in r:
            return True
    return False


def build_final_report(per_file):
    results = []
    for entry in per_file:
        m = entry["overall"]
        p = Path(entry["input"])
        ratio_match = re.search(r'_r(\d+)\.jsonl$', p.name)
        if not ratio_match:
            continue
        ratio = int(ratio_match.group(1)) / 100

        ver = m["verification"]
        eff = m["efficiency"]
        sp = m["system_performance"]

        results.append({
            "ratio": ratio,
            "system_accuracy": sp["system_accuracy_after_verification"],
            "generator_accuracy": sp["generator_accuracy"],
            "tokens_per_correct": round(eff["tokens_per_correct"]),
            "accept_rate": ver["accept_rate"],
            "error_detection_rate": ver["error_detection_rate"],
        })

    results.sort(key=lambda x: x["ratio"])
    ratios = [r["ratio"] for r in results]

    curves = {
        "system_accuracy_vs_ratio":       [[r["ratio"], r["system_accuracy"]]   for r in results],
        "generator_accuracy_vs_ratio":     [[r["ratio"], r["generator_accuracy"]] for r in results],
        "tokens_per_correct_vs_ratio":     [[r["ratio"], r["tokens_per_correct"]] for r in results],
    }

    fnr_map = {}
    for entry in per_file:
        p = Path(entry["input"])
        ratio_match = re.search(r'_r(\d+)\.jsonl$', p.name)
        if ratio_match:
            fnr_map[int(ratio_match.group(1)) / 100] = entry["overall"]["verification"]["false_negative_rate"]
    fnr_curve = sorted(
        [[r, fnr_map[r]] for r in fnr_map if r in {res["ratio"] for res in results}],
        key=lambda x: x[0],
    )
    if fnr_curve:
        curves["false_negative_rate_vs_ratio"] = fnr_curve

    best = max(results, key=lambda r: r["system_accuracy"])
    r_low = min(ratios) if len(ratios) > 1 else None
    r_high = max(ratios)

    headline = {
        "best_ratio": best["ratio"],
        "best_system_accuracy": best["system_accuracy"],
    }
    if r_low is not None:
        ratio_low_data = next((r for r in results if r["ratio"] == r_low), None)
        ratio_high_data = next((r for r in results if r["ratio"] == r_high), None)
        if ratio_low_data and ratio_high_data:
            headline[f"gain_{r_low}_to_{r_high}"] = round(ratio_high_data["system_accuracy"] - ratio_low_data["system_accuracy"], 4)

    dataset = "unknown"
    if per_file:
        name = Path(per_file[0]["input"]).stem
        dataset = name.split("_")[0] if name else "unknown"

    return {
        "meta": {
            "dataset": dataset,
            "model": "gpt-4o-mini",
            "sweep_dimension": "generator_ratio",
            "ratios": ratios,
            "n_total": sum(r["overall"]["data_quality"]["n_valid"] for r in per_file),
        },
        "results": results,
        "curves": curves,
        "headline_metrics": headline,
    }


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input",
        help="Single JSONL file to analyze. If this points to a directory, all JSONLs in that directory are analyzed.",
    )
    src.add_argument("--input-dir", help="Directory containing JSONL files to analyze")
    src.add_argument("--inputs", nargs="+", help="Explicit list of JSONL files")

    parser.add_argument(
        "--output",
        default="summary.json",
        help="Output path. If this is a directory (or has no file extension), writes summary.json and summary.jsonl inside it.",
    )
    parser.add_argument("--final-report", action="store_true", help="Generate a final sweep report from all ratio runs")
    parser.add_argument("--group-by", default="budget.generator_ratio")
    parser.add_argument("--recursive", action="store_true", help="Recurse when using --input-dir")
    args = parser.parse_args()

    input_files: list[Path]
    if args.inputs:
        input_files = [Path(p) for p in args.inputs]
    elif args.input_dir:
        input_files = _iter_jsonl_files(Path(args.input_dir), recursive=bool(args.recursive))
    else:
        p = Path(args.input)
        if p.exists() and p.is_dir():
            input_files = _iter_jsonl_files(p, recursive=bool(args.recursive))
        else:
            input_files = [p]

    per_file: list[dict] = []
    skipped: list[dict] = []

    for p in input_files:
        try:
            rows = list(load_jsonl(str(p)))
        except Exception as exc:
            skipped.append({"input": str(p), "reason": "load_failed", "error": repr(exc)})
            continue

        dict_rows = [r for r in rows if isinstance(r, dict)]
        if not _looks_like_pilot_output(dict_rows):
            skipped.append({"input": str(p), "reason": "missing_verifier_decision"})
            continue

        overall = compute_metrics(dict_rows)
        per_file.append(
            {
                "input": str(p),
                "overall": overall,
            }
        )

    result = {
        "inputs": [str(p) for p in input_files],
        "n_inputs": len(input_files),
        "n_analyzed": len(per_file),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "results": per_file,
    }

    out_arg = Path(args.output)
    # If --output is a directory OR looks like a directory (no suffix), place
    # outputs inside it.
    if (out_arg.exists() and out_arg.is_dir()) or (out_arg.suffix == ""):
        out_dir = out_arg
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "summary.json"
    else:
        out_path = out_arg
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Also write a compact JSONL (one line per input file).
    jsonl_path = out_path.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in per_file:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved:", str(out_path))
    print("Saved:", str(jsonl_path))

    if args.final_report:
        report = build_final_report(per_file)
        report_path = out_path.parent / "final_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Saved:", str(report_path))


if __name__ == "__main__":
    main()
