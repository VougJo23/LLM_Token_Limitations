import json
import argparse
from pathlib import Path
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt

from src.utils.io import save_jsonl


def load_points(input_dir, recursive=False):
    points = []

    files = (
        input_dir.rglob("*_summary.json")
        if recursive
        else input_dir.glob("*_summary.json")
    )

    for path in sorted(files):
        summary = json.loads(path.read_text(encoding="utf-8"))

        budget = summary.get("budget", {})

        points.append({
            "dataset": summary.get("dataset"),
            "model": summary.get("model"),

            "generator_ratio": float(budget.get("generator_ratio", 0)),
            "generator_max_tokens": int(budget.get("generator_max_tokens", 0)),
            "verifier_max_tokens": int(budget.get("verifier_max_tokens", 0)),

            "generator_accuracy": float(summary.get("generator_accuracy", 0)),
            "verifier_accuracy": float(summary.get("verifier_accuracy", 0)),

            "false_positive_rate": float(summary.get("false_positive_rate", 0)),
            "false_negative_rate": float(summary.get("false_negative_rate", 0)),

            "system_level_accuracy": float(summary.get("system_level_accuracy", 0)),

            "generator_truncation_rate": float(
                summary.get("generator_truncation_rate", 0)
            ),

            "summary_path": str(path),
            "output_jsonl": summary.get("output"),
        })

    return points


def _safe_mean(values):
    values = list(values)
    return float(mean(values)) if values else 0.0


def _summarize(points):
    metrics = [
        "generator_accuracy",
        "verifier_accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "system_level_accuracy",
        "generator_truncation_rate",
    ]

    overall = {m: _safe_mean(p.get(m, 0.0) for p in points) for m in metrics}

    by_vb = defaultdict(list)
    for p in points:
        by_vb[int(p.get("verifier_max_tokens", 0) or 0)].append(p)

    by_verifier_budget = {
        str(vb): {m: _safe_mean(p.get(m, 0.0) for p in rows) for m in metrics}
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
        xs = [v[x_key] for v in vals]
        ys = [v[y_key] for v in vals]

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
    print(f"Wrote: {out_vacc}")
    print(f"Wrote: {out_sys}")
    print(f"Wrote: {out_gacc}")
    print(f"Wrote: {out_gtr}")


if __name__ == "__main__":
    main()


