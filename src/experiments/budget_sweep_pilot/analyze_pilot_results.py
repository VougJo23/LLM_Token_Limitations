import json
import argparse
from collections import defaultdict
from statistics import mean

from src.utils.io import load_jsonl

def compute_metrics(rows):
    valid = [r for r in rows if "error" not in r]
    decided = [r for r in valid if r.get("verifier_decision") is not None]

    total_valid = len(valid)
    total_decided = len(decided)

    def is_correct(r):
        return r.get("generator_correct") or r.get("actual_correctness")

    correct_accepts = sum(
        1 for r in decided
        if is_correct(r) is True and r.get("verifier_decision") is True
    )

    false_rejects = sum(
        1 for r in decided
        if is_correct(r) is True and r.get("verifier_decision") is False
    )

    correct_rejections = sum(
        1 for r in decided
        if is_correct(r) is False and r.get("verifier_decision") is False
    )

    lazy_accepts = sum(
        1 for r in decided
        if is_correct(r) is False and r.get("verifier_decision") is True
    )

    fpr_denom = lazy_accepts + correct_rejections
    fnr_denom = false_rejects + correct_accepts

    false_positive_rate = lazy_accepts / fpr_denom if fpr_denom else 0.0
    false_negative_rate = false_rejects / fnr_denom if fnr_denom else 0.0

    system_level_accuracy = (
        (correct_accepts + correct_rejections) / total_decided
        if total_decided else 0.0
    )

    effective_acceptance_rate = total_decided / total_valid if total_valid else 0.0

    def avg_bool(key):
        return mean([bool(r.get(key)) for r in valid]) if valid else 0.0

    def avg_nested(path1, path2):
        vals = []
        for r in valid:
            if path1:
                vals.append(r.get(path1, {}).get(path2, 0))
            else:
                vals.append(r.get(path2, 0))
        return mean(vals) if vals else 0.0

    return {
        "n_total": len(rows),
        "n_valid": total_valid,
        "n_decided": total_decided,

        "generator_accuracy": avg_bool("generator_correct"),
        "verifier_accuracy": avg_bool("verifier_correct"),

        "lazy_accept_rate": avg_bool("verifier_lazy_accept"),
        "abstain_rate": avg_bool("verifier_abstained"),

        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,

        "system_level_accuracy": system_level_accuracy,
        "effective_acceptance_rate": effective_acceptance_rate,

        "generator_truncation_rate": avg_bool("truncated"),
        "verifier_truncation_rate": avg_bool("verifier_truncated"),

        "avg_generator_total_tokens": avg_nested("usage", "generator_total_tokens"),
        "avg_verifier_total_tokens": avg_nested("usage", "verifier_total_tokens"),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="summary.json")
    parser.add_argument("--group-by", default="budget.generator_ratio")
    args = parser.parse_args()

    rows = list(load_jsonl(args.input))

    overall = compute_metrics(rows)
    grouped = group_and_summarize(rows, args.group_by)

    result = {
        "input": args.input,
        "overall": overall,
        "grouped": grouped,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    jsonl_path = args.output.replace(".json", ".jsonl")

    with open(jsonl_path, "w") as f:
        for k, v in grouped.items():
            f.write(json.dumps({"group": k, "metrics": v}) + "\n")

    print("Saved:", args.output)
    print("Saved:", jsonl_path)


if __name__ == "__main__":
    main()
