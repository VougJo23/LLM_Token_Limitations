import json
import argparse
from pathlib import Path
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt

from src.utils.io import save_jsonl

# + analyze how many tokens were use to not truncate per difficulty

_EXPECTED_METRIC_KEYS = (
    "generator_accuracy",
    "verifier_accuracy",
    "false_positive_rate",
    "error_detection_rate",
    "false_negative_rate",
    "system_level_accuracy",
    "generator_truncation_rate",
)


def _to_float(x, default=0.0):
    if x is None:
        return float(default)
    if isinstance(x, bool):
        return float(x)
    try:
        return float(x)
    except Exception:
        return float(default)


def load_points(input_dir, recursive=False):
    points = []

    files = (
        input_dir.rglob("*_summary.json")
        if recursive
        else input_dir.glob("*_summary.json")
    )

    for path in sorted(files):

        if path.name in {"analysis_summary.json"}:
            continue

        summary = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(summary, dict):
            continue
        
        if not any(k in summary for k in _EXPECTED_METRIC_KEYS):
            continue

        budget = summary.get("budget", {})
        if not isinstance(budget, dict):
            continue

        dataset = summary.get("dataset")
        model = summary.get("model")
        if dataset is None or model is None:
            continue

        point = dict(summary)
        point["dataset"] = dataset
        point["model"] = model
        point["summary_path"] = str(path)
        point["output_jsonl"] = summary.get("output")

        point["generator_ratio"] = _to_float(budget.get("generator_ratio", 0), default=0.0)
        point["generator_max_tokens"] = int(budget.get("generator_max_tokens", 0) or 0)
        point["verifier_max_tokens"] = int(budget.get("verifier_max_tokens", 0) or 0)

        if "error_detection_rate" not in point and "false_positive_rate" in point:
            point["error_detection_rate"] = 1.0 - _to_float(point.get("false_positive_rate"), default=0.0)

        points.append(point)

    return points


def _safe_mean(values):
    values = list(values)
    return float(mean(values)) if values else 0.0


def _summarize(points):
    metrics = [
        "generator_accuracy",
        "verifier_accuracy",
        "false_positive_rate",
        "error_detection_rate",
        "false_negative_rate",
        "system_level_accuracy",
        "generator_truncation_rate",
    ]

    overall = {m: _safe_mean(_to_float(p.get(m), default=0.0) for p in points) for m in metrics}

    by_vb = defaultdict(list)
    for p in points:
        by_vb[int(p.get("verifier_max_tokens", 0) or 0)].append(p)

    by_verifier_budget = {
        str(vb): {m: _safe_mean(_to_float(p.get(m), default=0.0) for p in rows) for m in metrics}
        | {"n": len(rows)}
        for vb, rows in sorted(by_vb.items(), key=lambda kv: kv[0])
        if vb > 0
    }

    return {"overall": overall, "by_verifier_max_tokens": by_verifier_budget}

def compute_rates(rows):
    total = len(rows)

    correct_accepts = 0
    correct_rejections = 0
    false_accepts = 0
    false_rejects = 0

    for r in rows:
        gold = r["actual_correctness"]
        pred = r["verifier_decision"]

        if gold and pred:
            correct_accepts += 1

        elif gold and not pred:
            false_rejects += 1

        elif not gold and pred:
            false_accepts += 1

        else:
            correct_rejections += 1

    verifier_accuracy = (
        (correct_accepts + correct_rejections) / total
        if total else 0
    )

    fpr = (
        false_accepts / (false_accepts + correct_rejections)
        if (false_accepts + correct_rejections) > 0 else 0
    )

    fnr = (
        false_rejects / (false_rejects + correct_accepts)
        if (false_rejects + correct_accepts) > 0 else 0
    )

    return {
        "verifier_accuracy": verifier_accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }
    
    
def plot_metric(points, x_key, y_key, title, out_path):
    grouped = defaultdict(list)

    for p in points:
        grouped[p["model"]].append(p)

    plt.figure(figsize=(7, 5))

    for model, vals in grouped.items():
        xs = [_to_float(v.get(x_key), default=0.0) for v in vals]
        ys = [_to_float(v.get(y_key), default=0.0) for v in vals]

        plt.scatter(xs, ys, label=model)

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)

    plt.legend()
    plt.grid(True)

    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/experiments/pilot")
    parser.add_argument("--output-dir", default="data/experiments/pilot/analysis")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(input_dir, recursive=bool(args.recursive))
    summary = _summarize(points)

    # Save data
    out_points_json = output_dir / "summary_points.json"
    out_points_json.write_text(
        json.dumps(points, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out_points_jsonl = output_dir / "summary_points.jsonl"
    save_jsonl(points, out_points_jsonl)

    out_summary_json = output_dir / "analysis_summary.json"
    out_summary_json.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "recursive": bool(args.recursive),
                "n_summaries": len(points),
                "ratio_summaries": points,
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Plots
    out_fpr = output_dir / "false_positive_rate_vs_verifier_budget.png"
    plot_metric(points, "verifier_max_tokens", "false_positive_rate", "FPR vs verifier budget", out_fpr)

    out_fnr = output_dir / "false_negative_rate_vs_verifier_budget.png"
    plot_metric(points, "verifier_max_tokens", "false_negative_rate", "FNR vs verifier budget", out_fnr)

    out_edr = output_dir / "error_detection_rate_vs_verifier_budget.png"
    plot_metric(points, "verifier_max_tokens", "error_detection_rate", "EDR vs verifier budget", out_edr)

    out_vacc = output_dir / "verifier_accuracy_vs_verifier_budget.png"
    plot_metric(points, "verifier_max_tokens", "verifier_accuracy", "Verifier accuracy vs verifier budget", out_vacc)

    out_sys = output_dir / "system_level_accuracy_vs_verifier_budget.png"
    plot_metric(points, "verifier_max_tokens", "system_level_accuracy", "System accuracy vs verifier budget", out_sys)

    out_gacc = output_dir / "generator_accuracy_vs_generator_ratio.png"
    plot_metric(points, "generator_ratio", "generator_accuracy", "Generator accuracy vs generator ratio", out_gacc)

    out_gtr = output_dir / "generator_truncation_rate_vs_generator_ratio.png"
    plot_metric(points, "generator_ratio", "generator_truncation_rate", "Gen truncation vs generator ratio", out_gtr)

    print(f"Loaded {len(points)} summaries from: {input_dir} (recursive={bool(args.recursive)})")
    print(f"Wrote: {out_points_json}")
    print(f"Wrote: {out_points_jsonl}")
    print(f"Wrote: {out_summary_json}")
    print(f"Wrote: {out_fpr}")
    print(f"Wrote: {out_fnr}")
    print(f"Wrote: {out_edr}")
    print(f"Wrote: {out_vacc}")
    print(f"Wrote: {out_sys}")
    print(f"Wrote: {out_gacc}")
    print(f"Wrote: {out_gtr}")


if __name__ == "__main__":
    main()


