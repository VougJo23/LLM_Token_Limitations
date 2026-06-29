"""
Unified Paper Artifact Generator
Usage: python paper_results.py --files path/to/report1.json path/to/report2.json --out ./paper_artifacts
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_files(files):
    # Change: Store as a list of dicts with a 'model' label
    combined_data = []
    
    for file_path in files:
        path = Path(file_path)
        # Infer model name from path (e.g., ".../openai/..." -> "openai")
        model_name = "openai" if "openai" in str(path).lower() else "qwen"
        
        with open(path, "r") as f:
            data = json.load(f)
            by_diff = data.get("by_ratio_and_difficulty", {})
            for ratio, stats in data["overall_by_ratio"].items():
                stats["ratio"] = ratio
                stats["model"] = model_name
                stats["by_difficulty"] = by_diff.get(ratio, {})
                combined_data.append(stats)
    return combined_data

def flatten_for_export(data_list):
    rows = []
    for stats in data_list:
        ad = stats["all_decided"]
        perf = ad["performance"]
        sdt = ad["signal_detection"]
        reasoning = ad["reasoning"]
        
        trunc = ad["truncation"]
        row = {
            "model": stats["model"],
            "ratio": float(stats["ratio"]),
            "gen_acc": perf["generator_accuracy"],
            "ver_acc": perf["verifier_accuracy"],
            "sys_acc": perf["system_accuracy"],
            "fnr": perf["false_negative_rate"],
            "fpr": perf["false_positive_rate"],
            "process_outcome_mismatch_rate": reasoning["process_outcome_mismatch_rate"],
            "edr": perf["error_detection_rate"],
            "d_prime": sdt["d_prime"],
            "c_score": sdt["criterion"],
            "c_lower": sdt["criterion_ci95"][0] if sdt["criterion_ci95"] else None,
            "c_upper": sdt["criterion_ci95"][1] if sdt["criterion_ci95"] else None,
            "mean_gen_len": stats["confounds"]["generator_trace_len_mean"],
            "n_valid": ad["data_quality"]["n_valid"],
            "n_trunc": trunc["total_truncated"],
            "n_no_verdict": trunc["no_verdict"],
            "n_decided": ad["data_quality"]["n_decided"],
            "gen_trunc_total": trunc.get("gen_truncated_total", 0),
            "gen_trunc_reject": trunc.get("gen_truncated_reject", 0),
            "gen_trunc_accept": trunc.get("gen_truncated_accept", 0),
            "gen_trunc_no_verdict": trunc.get("gen_truncated_no_verdict", 0)
        }
        rows.append(row)
    return pd.DataFrame(rows)

def _model_data(merged_data, model):
    mdata = [d for d in merged_data if d["model"] == model]
    mdata.sort(key=lambda d: float(d["ratio"]))
    ratios = [float(d["ratio"]) for d in mdata]
    c_scores = [d["all_decided"]["signal_detection"]["criterion"] for d in mdata]
    d_primes = [d["all_decided"]["signal_detection"]["d_prime"] for d in mdata]
    return ratios, c_scores, d_primes

def _extract_by_difficulty(merged_data, model):
    result = {}
    for d in merged_data:
        if d["model"] != model:
            continue
        ratio = float(d["ratio"])
        bd = d.get("by_difficulty", {})
        result[ratio] = {}
        for diff in ["easy", "medium", "hard"]:
            g = bd.get(diff)
            if g:
                sd = g["all_decided"]["signal_detection"]
                result[ratio][diff] = {
                    "d_prime": sd["d_prime"],
                    "c": sd["criterion"],
                }
            else:
                result[ratio][diff] = None
    return result


def plot_criterion_trajectory(merged_data, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    model_colors = {"openai": "#1f77b4", "qwen": "#ff7f0e"}

    for model in ("openai", "qwen"):
        ratios, c_scores, _ = _model_data(merged_data, model)
        clean_r, clean_c = zip(*[(r, c) for r, c in zip(ratios, c_scores) if r <= 0.75])
        high_r, high_c = zip(*[(r, c) for r, c in zip(ratios, c_scores) if r > 0.75])

        if clean_r:
            ax.plot(clean_r, clean_c, "-", color=model_colors[model],
                    label=f"{model}", linewidth=1.8)
        if high_r:
            ax.plot(high_r, high_c, "--", color="gray", linewidth=1.8)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xlabel("Generator ratio R")
    ax.set_ylabel("Criterion c")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_criterion_trajectory.png", dpi=200)
    plt.close()

def plot_sensitivity_leniency_dual(merged_data, out_dir):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()
    model_colors = {"openai": "#1f77b4", "qwen": "#ff7f0e"}
    metric_styles = {"d_prime": "-", "c": "--"}

    for model in ("openai", "qwen"):
        ratios, c_scores, d_primes = _model_data(merged_data, model)
        clean_r, clean_d, clean_c = zip(*[(r, d, c) for r, d, c in zip(ratios, d_primes, c_scores) if r <= 0.75])

        ax1.plot(clean_r, clean_d, metric_styles["d_prime"], color=model_colors[model],
                 label=f"d' ({model})", linewidth=1.8)
        ax2.plot(clean_r, clean_c, metric_styles["c"], color=model_colors[model],
                 label=f"c ({model})", linewidth=1.8)

    ax1.set_xlabel("Generator ratio R (clean range)")
    ax1.set_ylabel("d'", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylabel("c", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_sensitivity_leniency_dual.png", dpi=200)
    plt.close()

def plot_truncation_vs_fpr(merged_data, out_dir):
    model_colors = {"openai": "#1f77b4", "qwen": "#ff7f0e"}
    metric_styles = {"trunc_rate": "-", "fpr": "--"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for model, color in model_colors.items():
        mdata = [d for d in merged_data if d["model"] == model]
        mdata.sort(key=lambda d: float(d["ratio"]))
        ratios = [float(d["ratio"]) for d in mdata]
        n_valid = [max(d["all_decided"]["data_quality"]["n_valid"], 1) for d in mdata]
        trunc_rate = [d["all_decided"]["truncation"]["total_truncated"] / nv
                      for d, nv in zip(mdata, n_valid)]
        fpr = [d["all_decided"]["performance"]["false_positive_rate"] for d in mdata]

        ax.plot(ratios, trunc_rate, metric_styles["trunc_rate"], color=color,
                label=f"Trunc rate ({model})", linewidth=1.8)
        ax.plot(ratios, fpr, metric_styles["fpr"], color=color,
                label=f"FPR ({model})", linewidth=1.8)

    r90_data = [d for d in merged_data if abs(float(d["ratio"]) - 0.90) < 0.01]
    if r90_data:
        r90 = r90_data[0]
        nv = max(r90["all_decided"]["data_quality"]["n_valid"], 1)
        r90_trunc = r90["all_decided"]["truncation"]["total_truncated"] / nv
        ax.annotate("Truncation cliff", xy=(0.90, r90_trunc),
                    xytext=(0.70, r90_trunc + 0.15),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=10, color="gray")

    ax.set_xlabel("Generator ratio R")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "truncation_vs_fpr.png", dpi=200)
    plt.close()


def _bar_plot_by_difficulty(merged_data, out_dir, metric_key, ylabel, fname):
    target_ratios = [0.30, 0.45, 0.60, 0.75]
    difficulties = ["easy", "medium", "hard"]
    diff_labels = ["Easy", "Medium", "Hard"]
    models = ["openai", "qwen"]
    model_labels = {"openai": "OpenAI", "qwen": "Qwen"}
    na_specs = {("openai", "easy", 0.30)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(target_ratios)))

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        data = _extract_by_difficulty(merged_data, model)

        n_diffs = len(difficulties)
        n_ratios = len(target_ratios)
        bar_width = 0.2
        x = np.arange(n_diffs)

        for ri, ratio in enumerate(target_ratios):
            offset = (ri - (n_ratios - 1) / 2) * bar_width
            values = []
            for di, diff in enumerate(difficulties):
                key = (model, diff, ratio)
                val = None
                if ratio in data and data[ratio] and data[ratio][diff]:
                    val = data[ratio][diff][metric_key]
                if key in na_specs:
                    val = None
                values.append(val)

            valid_x = [x[di] + offset for di in range(n_diffs) if values[di] is not None]
            valid_y = [values[di] for di in range(n_diffs) if values[di] is not None]
            if valid_x:
                ax.bar(valid_x, valid_y, bar_width, color=colors[ri],
                       label="R=%.2f" % ratio if ax_idx == 0 else None)

            for di in range(n_diffs):
                if values[di] is None and (model, difficulties[di], ratio) in na_specs:
                    ax.text(x[di] + offset, 0, "N/A", ha="center", va="bottom",
                            fontsize=8, color="gray", fontweight="bold")

        ax.set_title(model_labels[model])
        ax.set_xticks(x)
        ax.set_xticklabels(diff_labels)
        ax.set_xlabel("Difficulty")
        if ax_idx == 0:
            ax.set_ylabel(ylabel)
        if ax_idx == 1:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=200)
    plt.close()


def plot_dprime_by_difficulty(merged_data, out_dir):
    _bar_plot_by_difficulty(merged_data, out_dir, "d_prime", "d'", "fig_dprime_by_difficulty.png")


def plot_criterion_by_difficulty(merged_data, out_dir):
    _bar_plot_by_difficulty(merged_data, out_dir, "c", "c", "fig_criterion_by_difficulty.png")


def main():
    parser = argparse.ArgumentParser()
    # Accept a list of specific files
    parser.add_argument("--files", nargs='+', required=True, help="List of specific master_report files")
    parser.add_argument("--out", default="paper_artifacts", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_data = load_files(args.files)
    if not merged_data:
        print("No data loaded. Check your file paths.")
        return

    df = flatten_for_export(merged_data)

    value_columns = [
        'gen_acc', 'ver_acc', #'sys_acc', 
        'fnr', 'fpr', 'process_outcome_mismatch_rate', 'edr', 
        'd_prime', 'c_score', 'c_lower', 'c_upper', 
        'mean_gen_len', 'n_valid', 'n_trunc', 'n_no_verdict', 'n_decided',
        'gen_trunc_total', 'gen_trunc_reject', 'gen_trunc_accept', 'gen_trunc_no_verdict'
    ]
    df_wide = df.pivot(index='ratio', columns='model', values=value_columns)
    df_wide.columns = [f"{col[0]}_{col[1]}" for col in df_wide.columns]
    df_wide = df_wide.reset_index()

    df_wide[[
        "ratio", 
        "c_score_openai", "c_score_qwen", 
        "c_lower_openai", "c_lower_qwen", 
        "c_upper_openai", "c_upper_qwen", 
        "d_prime_openai", "d_prime_qwen"
    ]].to_csv(out_dir / "plot_criterion_leniency.csv", index=False)
    
    # Leniency Overview (LaTeX)
    df_wide[[
        "ratio", 
        "fnr_openai", "fnr_qwen",
        "fpr_openai", "fpr_qwen", 
        "process_outcome_mismatch_rate_openai", "process_outcome_mismatch_rate_qwen", 
        "edr_openai", "edr_qwen", 
        "d_prime_openai", "d_prime_qwen", 
        "c_score_openai", "c_score_qwen"
    ]].to_latex(out_dir / "table_leniency_overview.tex", index=False, float_format="%.3f")
    
    # Starvation Floor (LaTeX)
    df_wide[[
        "ratio", 
        "n_valid_openai", "n_valid_qwen", 
        "n_trunc_openai", "n_trunc_qwen", 
        "n_no_verdict_openai", "n_no_verdict_qwen", 
        "n_decided_openai", "n_decided_qwen"
    ]].to_latex(out_dir / "table_starvation_floor.tex", index=False)
    
    # Confound Tracking (LaTeX)
    df_wide[[
        "ratio", 
        "gen_acc_openai", "gen_acc_qwen", 
        "ver_acc_openai", "ver_acc_qwen", 
        #"sys_acc_openai", "sys_acc_qwen", 
        "mean_gen_len_openai", "mean_gen_len_qwen"
    ]].to_latex(out_dir / "table_confound_tracking.tex", index=False, float_format="%.2f")
    
    # Generator Truncation Breakdown (LaTeX)
    df_wide[[
        "ratio",
        "gen_trunc_total_openai", "gen_trunc_total_qwen",
        "gen_trunc_reject_openai", "gen_trunc_reject_qwen",
        "gen_trunc_accept_openai", "gen_trunc_accept_qwen",
        "gen_trunc_no_verdict_openai", "gen_trunc_no_verdict_qwen"
    ]].to_latex(out_dir / "table_gen_truncation.tex", index=False)
    
    plot_truncation_vs_fpr(merged_data, out_dir)
    plot_criterion_trajectory(merged_data, out_dir)
    plot_sensitivity_leniency_dual(merged_data, out_dir)
    plot_dprime_by_difficulty(merged_data, out_dir)
    plot_criterion_by_difficulty(merged_data, out_dir)
    print(f"Generated wide-format artifacts in '{out_dir}/'")

if __name__ == "__main__":
    main()
