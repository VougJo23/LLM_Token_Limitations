"""
Unified Paper Artifact Generator
Usage: python paper_results.py --files path/to/report1.json path/to/report2.json --out ./paper_artifacts
"""
import json
import argparse
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
            for ratio, stats in data["overall_by_ratio"].items():
                stats["ratio"] = ratio
                stats["model"] = model_name
                combined_data.append(stats)
    return combined_data

def flatten_for_export(data_list):
    rows = []
    for stats in data_list:
        ad = stats["all_decided"]
        perf = ad["performance"]
        sdt = ad["signal_detection"]
        
        trunc = ad["truncation"]
        row = {
            "model": stats["model"],
            "ratio": float(stats["ratio"]),
            "gen_acc": perf["generator_accuracy"],
            "ver_acc": perf["verifier_accuracy"],
            "sys_acc": perf["system_accuracy"],
            "fpr": perf["false_positive_rate"],
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

def main():
    parser = argparse.ArgumentParser()
    # Accept a list of specific files
    parser.add_argument("--files", nargs='+', required=True, help="List of specific master_report files")
    parser.add_argument("--out", default="paper_artifacts", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load exact files
    merged_data = load_files(args.files)
    if not merged_data:
        print("❌ No data loaded. Check your file paths.")
        return

    # 2. Flatten data (Unchanged logic)
    df = flatten_for_export(merged_data)

    # 3. Pivot into "Wide" side-by-side format
    value_columns = [
        'gen_acc', 'ver_acc', 'sys_acc', 'fpr', 'edr', 
        'd_prime', 'c_score', 'c_lower', 'c_upper', 
        'mean_gen_len', 'n_valid', 'n_trunc', 'n_no_verdict', 'n_decided',
        'gen_trunc_total', 'gen_trunc_reject', 'gen_trunc_accept', 'gen_trunc_no_verdict'
    ]
    df_wide = df.pivot(index='ratio', columns='model', values=value_columns)
    
    # Flatten the multi-index columns (e.g., ('edr', 'openai') -> 'edr_openai')
    df_wide.columns = [f"{col[0]}_{col[1]}" for col in df_wide.columns]
    df_wide = df_wide.reset_index()

    # 4. Export side-by-side artifacts
    # CSV Plot Data
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
        "fpr_openai", "fpr_qwen", 
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
        "sys_acc_openai", "sys_acc_qwen", 
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
    
    print(f"✅ Generated wide-format artifacts in '{out_dir}/'")

if __name__ == "__main__":
    main()
