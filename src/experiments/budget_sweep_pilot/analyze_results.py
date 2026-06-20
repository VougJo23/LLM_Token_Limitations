import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.stats import norm
from statistics import mean, median

def _as_bool(x):
    return x if x is True or x is False else None

def _gold(r):
    for key in ("actual_correctness", "generator_correct", "correct"):
        if key in r and _as_bool(r[key]) is not None:
            return _as_bool(r[key])
    return None

def _get_budget(r, key, default=None):
    b = r.get("budget", {})
    return b.get(key, r.get(key, default))

def deep_search(data, target_keys):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(k, str) and k.lower() in target_keys and v is not None:
                try: return float(v)
                except (ValueError, TypeError): pass
            if isinstance(v, (dict, list)):
                res = deep_search(v, target_keys)
                if res is not None: return res
    elif isinstance(data, list):
        for item in data:
            res = deep_search(item, target_keys)
            if res is not None: return res
    return None

def extract_ratio(filename):
    match = re.search(r'_r(\d+)', filename)
    return round(float(match.group(1)) / 100, 4) if match else 0.0

def _sdt_base(gold_arr, accept_arr):
    n_pos, n_neg = int(gold_arr.sum()), int((~gold_arr).sum())
    if n_pos == 0 or n_neg == 0:
        return None, None
    tp = int((gold_arr & accept_arr).sum())
    fp = int((~gold_arr & accept_arr).sum())
    h = (tp + 0.5) / (n_pos + 1.0)
    fa = (fp + 0.5) / (n_neg + 1.0)
    z_h, z_fa = float(norm.ppf(h)), float(norm.ppf(fa))
    return z_h - z_fa, -0.5 * (z_h + z_fa)

def compute_sdt(gold_arr, accept_arr, rng, n_boot=2000):
    """Signal Detection Theory with 95% Bootstrapped Confidence Intervals."""
    d_prime, criterion = _sdt_base(gold_arr, accept_arr)
    res = {
        "d_prime": d_prime,
        "criterion": criterion,
        "d_prime_ci95": None,
        "criterion_ci95": None
    }
    
    if rng is not None and d_prime is not None:
        n = len(gold_arr)
        d_vals, c_vals = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            dp, c = _sdt_base(gold_arr[idx], accept_arr[idx])
            if dp is not None:
                d_vals.append(dp)
                c_vals.append(c)
        if d_vals:
            res["d_prime_ci95"] = [float(np.percentile(d_vals, 2.5)), float(np.percentile(d_vals, 97.5))]
            res["criterion_ci95"] = [float(np.percentile(c_vals, 2.5)), float(np.percentile(c_vals, 97.5))]
            
    return res

def calculate_metrics(valid, decided, rng):

    gold_arr = np.array([_gold(r) for r in decided], dtype=bool)
    accept_arr = np.array([r["verifier_decision"] for r in decided], dtype=bool)

    correct_accepts = int((gold_arr & accept_arr).sum())
    false_rejects = int((gold_arr & ~accept_arr).sum())
    correct_rejections = int((~gold_arr & ~accept_arr).sum()) 
    lazy_accepts = int((~gold_arr & accept_arr).sum())

    fpr_denom = lazy_accepts + correct_rejections
    fnr_denom = false_rejects + correct_accepts
    accuracy = (correct_accepts + correct_rejections) / len(decided) if decided else 0.0

    gen_tokens = [r.get("completion_tokens", 0) for r in valid]
    trunc_missing_steps = sum(1 for r in valid if r.get("verifier_truncated") and not r.get("step_judgments"))
    trunc_no_verdict = sum(1 for r in valid if r.get("verifier_truncated") and r.get("step_judgments") and r.get("verifier_decision") is None)
    trunc_reason = sum(1 for r in valid if r.get("verifier_truncated") and r.get("step_judgments") and r.get("verifier_decision") is not None)

    # Process-outcome mismatch & answer-missing rates
    mismatch_count = 0
    answer_missing_count = 0
    for r in valid:
        all_steps_valid = r.get("total_steps", 0) > 0 and r.get("n_invalid_steps", 0) == 0
        answer_missing = r.get("truncated") or not r.get("predicted_answer")
        if answer_missing:
            answer_missing_count += 1
        if all_steps_valid and answer_missing:
            mismatch_count += 1

    # Generator truncation breakdown by verifier outcome
    gen_truncated = [r for r in valid if r.get("truncated")]
    n_gen_trunc = len(gen_truncated)
    gen_trunc_reject = sum(1 for r in gen_truncated if r.get("verifier_decision") is False)
    gen_trunc_accept = sum(1 for r in gen_truncated if r.get("verifier_decision") is True)
    gen_trunc_no_verdict = sum(1 for r in gen_truncated if r.get("verifier_decision") is None)

    return {
        "data_quality": {"n_valid": len(valid), "n_decided": len(decided)},
        "performance": {
            "generator_accuracy": mean([_gold(r) for r in valid]) if valid else 0.0,
            "system_accuracy": accuracy,
            "verifier_accuracy": accuracy,
            "error_detection_rate": correct_rejections / fpr_denom if fpr_denom else 0.0,
            "false_positive_rate": lazy_accepts / fpr_denom if fpr_denom else 0.0,
            "false_negative_rate": false_rejects / fnr_denom if fnr_denom else 0.0
        },
        "signal_detection": compute_sdt(gold_arr, accept_arr, rng),
        "tokens": {
            "mean_gen_tokens": mean(gen_tokens) if gen_tokens else 0,
            "median_gen_tokens": median(gen_tokens) if gen_tokens else 0,
        },
        "truncation": {
            "total_truncated": trunc_missing_steps + trunc_no_verdict + trunc_reason,
            "missing_steps": trunc_missing_steps,
            "no_verdict": trunc_no_verdict,
            "reason_cut_off": trunc_reason,
            "gen_truncated_total": n_gen_trunc,
            "gen_truncated_reject": gen_trunc_reject,
            "gen_truncated_accept": gen_trunc_accept,
            "gen_truncated_no_verdict": gen_trunc_no_verdict
        },
        "reasoning": {
            "process_outcome_mismatch_count": mismatch_count,
            "process_outcome_mismatch_rate": mismatch_count / len(valid) if valid else 0.0,
            "answer_missing_count": answer_missing_count,
            "answer_missing_rate": answer_missing_count / len(valid) if valid else 0.0
        }
    }

def analyze_group(rows, rng):
    """Splits group into 'All Decided' and 'Complete Only' to isolate truncation confounds."""
    valid = [r for r in rows if "error" not in r]
    decided = [r for r in valid if r.get("verifier_decision") is not None and _gold(r) is not None]
    complete = [r for r in decided if not r.get("verifier_truncated")]

    gen_lens = [r.get("completion_tokens", 0) for r in valid]
    ver_budgets = [_get_budget(r, "verifier_max_tokens", 0) for r in valid]

    return {
        "confounds": {
            "generator_trace_len_mean": mean(gen_lens) if gen_lens else 0,
            "verifier_max_tokens_median": median(ver_budgets) if ver_budgets else 0, 
        },
        "all_decided": calculate_metrics(valid, decided, rng),
        "complete_only": calculate_metrics(valid, complete, rng)
    }

def plot_metric(data, x_key, y_func, title, ylabel, out_path, ci_func=None):
    """Plotter with optional 95% Confidence Interval shading."""
    xs, ys, y_err_low, y_err_high = [], [], [], []
    for x_val, group_data in sorted(data.items()):
        val = y_func(group_data)
        if val is not None:
            xs.append(x_val)
            ys.append(val)
            if ci_func:
                ci = ci_func(group_data)
                if ci and ci[0] is not None:
                    y_err_low.append(ci[0])          
                    y_err_high.append(ci[1])
    
    if not xs: return
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker="o", linewidth=1.6)
    
    # Add shaded CI region
    if ci_func and len(y_err_low) == len(xs):
        plt.fill_between(xs, y_err_low, y_err_high, alpha=0.2, color="C0", label="95% CI")
        plt.legend()
        
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with raw jsonl files")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"   
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RNG for rigorous reproducible bootstrapping
    rng = np.random.default_rng(42)

    summary_aurocs = {}
    auroc_keys = {"auroc", "auroc_margin", "roc_auc", "verifier_auroc"}
    for path in input_dir.rglob("*_summary.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            ratio = extract_ratio(path.name)
            found_auroc = deep_search(s, auroc_keys)
            if found_auroc is not None and not np.isnan(found_auroc):
                summary_aurocs[ratio] = found_auroc
        except Exception: pass

    rows_by_ratio = defaultdict(list)
    rows_by_vbudget = defaultdict(list)
    fps, fns = [], []

    for path in input_dir.rglob("*.jsonl"):
        if "summary" in path.name or "analysis" in str(path): continue
        ratio = extract_ratio(path.name)
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    r["_source_path"] = str(path)
                    rows_by_ratio[ratio].append(r)
                    v_budget = _get_budget(r, "verifier_max_tokens")
                    if v_budget is not None:
                        rows_by_vbudget[int(v_budget)].append(r)
                    gold, pred = _gold(r), _as_bool(r.get("verifier_decision"))
                    if gold is not None and pred is not None:
                        if pred is True and gold is False: fps.append(r)
                        if pred is False and gold is True: fns.append(r)
                except json.JSONDecodeError: pass  

    print("Running rigorous metric calculations and bootstrapping (this may take a few seconds)...")
    report = {"overall_by_ratio": {}, "overall_by_vbudget": {}, "by_ratio_and_difficulty": {}}
    
    for ratio, rows in sorted(rows_by_ratio.items()):
        group_stats = analyze_group(rows, rng)
        group_stats["all_decided"]["signal_detection"]["auroc"] = summary_aurocs.get(ratio, None)
        group_stats["all_decided"]["signal_detection"]["auroc_source"] = "summary_file" if ratio in summary_aurocs else "missing"
        report["overall_by_ratio"][ratio] = group_stats
        
        diffs = defaultdict(list)
        for r in rows: diffs[_get_budget(r, "difficulty", "unknown")].append(r)
        report["by_ratio_and_difficulty"][ratio] = {d: analyze_group(rs, rng) for d, rs in diffs.items()}

    for vb, rows in sorted(rows_by_vbudget.items()):
        report["overall_by_vbudget"][vb] = analyze_group(rows, rng)

    (out_dir / "master_report.json").write_text(json.dumps(report, indent=2))
    
    with open(out_dir / "false_positives.jsonl", "w", encoding="utf-8") as f:
        for r in fps: f.write(json.dumps(r) + "\n")
    with open(out_dir / "false_negatives.jsonl", "w", encoding="utf-8") as f:
        for r in fns: f.write(json.dumps(r) + "\n")

    plot_data_ratio = report["overall_by_ratio"]
    plot_data_vratio = {round(1.0 - r, 4): data for r, data in plot_data_ratio.items()}

    # 5. Generate Publication Plots
    plot_data_ratio = report["overall_by_ratio"]
    plot_data_vratio = {round(1.0 - r, 4): data for r, data in plot_data_ratio.items()}

    # Plots grouped by Generator Ratio
    plot_metric(plot_data_ratio, "Generator Ratio", lambda d: d["all_decided"]["performance"]["system_accuracy"], "System Accuracy vs Gen Ratio", "Accuracy", out_dir / "sys_acc_vs_gen_ratio.png")
    plot_metric(plot_data_ratio, "Generator Ratio", lambda d: d["all_decided"]["performance"]["generator_accuracy"], "Generator Accuracy vs Gen Ratio", "Accuracy", out_dir / "gen_acc_vs_gen_ratio.png")
    plot_metric(plot_data_ratio, "Generator Ratio", lambda d: d["all_decided"]["truncation"]["total_truncated"] / d["all_decided"]["data_quality"]["n_valid"] if d["all_decided"]["data_quality"]["n_valid"] else 0, "Truncation Rate vs Gen Ratio", "Truncation Rate", out_dir / "truncation_vs_gen_ratio.png")
    plot_metric(plot_data_ratio, "Generator Ratio", lambda d: d["all_decided"]["signal_detection"]["d_prime"], "Verifier d' vs Gen Ratio", "d'", out_dir / "dprime_vs_gen_ratio.png", ci_func=lambda d: d["all_decided"]["signal_detection"]["d_prime_ci95"])
    
    # Plots grouped by Verifier Ratio
    plot_metric(plot_data_vratio, "Verifier Ratio (1 - Gen Ratio)", lambda d: d["all_decided"]["performance"]["error_detection_rate"], "EDR vs Verifier Ratio", "Error Detection Rate", out_dir / "edr_vs_vratio.png")
    plot_metric(plot_data_vratio, "Verifier Ratio (1 - Gen Ratio)", lambda d: d["all_decided"]["performance"]["false_positive_rate"], "FPR vs Verifier Ratio", "False Positive Rate", out_dir / "fpr_vs_vratio.png")
    plot_metric(plot_data_vratio, "Verifier Ratio (1 - Gen Ratio)", lambda d: d["all_decided"]["performance"]["false_negative_rate"], "FNR vs Verifier Ratio", "False Negative Rate", out_dir / "fnr_vs_vratio.png")

    print(f"Analysis Results saved to {out_dir}")
    
if __name__ == "__main__":
    main()
