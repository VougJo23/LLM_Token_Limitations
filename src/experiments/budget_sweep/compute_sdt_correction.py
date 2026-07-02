import json
import csv
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import norm

# Compute raw and corrected SDT

DATA_DIR = Path("data/experiments/budget_sweep/main")
OUT_DIR = DATA_DIR

RNG = np.random.default_rng(42)
N_BOOT = 2000


def _gold(row):
    for key in ("actual_correctness", "generator_correct", "correct"):
        val = row.get(key)
        if val is True or val is False:
            return val
    return None


def _sdt(gold_arr, accept_arr):
    n_pos = int(gold_arr.sum())
    n_neg = int((~gold_arr).sum())
    if n_pos == 0 or n_neg == 0:
        return None, None, None, None
    tp = int((gold_arr & accept_arr).sum())
    fp = int((~gold_arr & accept_arr).sum())
    h = (tp + 0.5) / (n_pos + 1.0)
    fa = (fp + 0.5) / (n_neg + 1.0)
    z_h = float(norm.ppf(h))
    z_fa = float(norm.ppf(fa))
    d = z_h - z_fa
    c = -0.5 * (z_h + z_fa)
    fpr = fp / n_neg if n_neg else 0.0
    fnr = (n_pos - tp) / n_pos if n_pos else 0.0
    return d, c, fpr, fnr


def _sdt_with_ci(gold_arr, accept_arr):
    d, c, fpr, fnr = _sdt(gold_arr, accept_arr)
    d_ci, c_ci = None, None
    if d is not None:
        n = len(gold_arr)
        d_vals, c_vals = [], []
        for _ in range(N_BOOT):
            idx = RNG.integers(0, n, n)
            g = gold_arr[idx]
            a = accept_arr[idx]
            dp, cp, _, _ = _sdt(g, a)
            if dp is not None:
                d_vals.append(dp)
                c_vals.append(cp)
        if d_vals:
            d_ci = [float(np.percentile(d_vals, 2.5)), float(np.percentile(d_vals, 97.5))]
            c_ci = [float(np.percentile(c_vals, 2.5)), float(np.percentile(c_vals, 97.5))]
    return {
        "d": d, "c": c, "fpr": fpr, "fnr": fnr,
        "d_ci95": d_ci, "c_ci95": c_ci,
        "n_pos": int(gold_arr.sum()), "n_neg": int((~gold_arr).sum()),
        "n_total": len(gold_arr),
    }


def load_all_traces():
    traces_by_key = defaultdict(list)
    for model in ("openai", "qwen"):
        model_dir = DATA_DIR / model
        for jsonl_path in sorted(model_dir.glob("gsm8k_r*.jsonl")):
            ratio_match = re.search(r"_r(\d+)", jsonl_path.name)
            ratio = round(float(ratio_match.group(1)) / 100, 2) if ratio_match else 0.0
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if "error" in r:
                        continue
                    r["_model"] = model
                    r["_ratio"] = ratio
                    traces_by_key[(model, ratio)].append(r)
    return traces_by_key


def compute():
    traces_by_key = load_all_traces()

    fp_records = []
    fn_records = []

    results = []

    for (model, ratio), traces in sorted(traces_by_key.items()):
        model_label = "OpenAI" if model == "openai" else "Qwen"
        qwen_col = "Qwen" if model == "qwen" else "OpenAI"

        decided = [r for r in traces if r.get("verifier_decision") is not None and _gold(r) is not None]

        gold_raw = np.array([_gold(r) for r in decided], dtype=bool)
        accept = np.array([r["verifier_decision"] for r in decided], dtype=bool)
        raw = _sdt_with_ci(gold_raw, accept)

        for r in decided:
            g = _gold(r)
            vd = r["verifier_decision"]
            if vd is True and g is False:
                fp_records.append({
                    "ratio": ratio, "model": model, "id": r.get("id", ""),
                    "process_outcome_mismatch": r.get("process_outcome_mismatch", False),
                    "truncated": r.get("truncated", False),
                    "difficulty": r.get("budget", {}).get("difficulty", ""),
                })
            if vd is False and g is True:
                fn_records.append({
                    "ratio": ratio, "model": model, "id": r.get("id", ""),
                    "process_outcome_mismatch": r.get("process_outcome_mismatch", False),
                    "truncated": r.get("truncated", False),
                    "difficulty": r.get("budget", {}).get("difficulty", ""),
                })

        #  corrected SDT
        gold_corr = gold_raw.copy()
        n_reclassified = 0
        for i, r in enumerate(decided):
            is_mismatch = r.get("process_outcome_mismatch", False) is True
            verifier_accepted = r["verifier_decision"] is True
            gold_is_false = not bool(gold_raw[i])
            if is_mismatch and verifier_accepted and gold_is_false:
                gold_corr[i] = True
                n_reclassified += 1

        corr = _sdt_with_ci(gold_corr, accept)

        results.append({
            "model": model, "ratio": ratio,
            "raw": raw, "corr": corr,
            "n_reclassified": n_reclassified,
        })

        print(f"{model_label} R={ratio}: reclassified {n_reclassified} traces "
              f"(d' {raw['d']:.3f} -> {corr['d']:.3f}, "
              f"c {raw['c']:.3f} -> {corr['c']:.3f}, "
              f"FPR {raw['fpr']:.3f} -> {corr['fpr']:.3f})")


    def write_sorted(records, filename):
        records.sort(key=lambda x: (x["ratio"], x["model"]))
        path = OUT_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(records)} records to {path}")

    write_sorted(fp_records, "false_positives_sorted.jsonl")
    write_sorted(fn_records, "false_negatives_sorted.jsonl")


    csv_path = OUT_DIR / "sdt_correction_values.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ratio", "d_o", "d_c", "c_o", "c_c",
                     "fpr_o", "fpr_c", "fnr_o", "fnr_c",
                     "n_reclassified", "n_pos_o", "n_neg_o",
                     "n_pos_c", "n_neg_c"])
        for res in results:
            r, c = res["raw"], res["corr"]
            w.writerow([res["model"], res["ratio"],
                        f"{r['d']:.3f}", f"{c['d']:.3f}",
                        f"{r['c']:.3f}", f"{c['c']:.3f}",
                        f"{r['fpr']:.3f}", f"{c['fpr']:.3f}",
                        f"{r['fnr']:.3f}", f"{c['fnr']:.3f}",
                        res["n_reclassified"],
                        r["n_pos"], r["n_neg"],
                        c["n_pos"], c["n_neg"]])
    print(f"Wrote CSV to {csv_path}")


    by_ratio = {}
    for res in results:
        r = res["ratio"]
        if r not in by_ratio:
            by_ratio[r] = {}
        by_ratio[r][res["model"]] = res

    sorted_ratios = sorted(by_ratio.keys())

    def fmt_val(v):
        if v is None:
            return "---"
        return f"{v:.2f}"

    lines = []
    lines.append(r"\begin{tabular}{l*{12}{r}}")
    lines.append(r"\toprule")
    lines.append(r"  \multirow{2}{*}{$R$} &"
                 r" \multicolumn{2}{c}{$d'_o$} &"
                 r" \multicolumn{2}{c}{$d'_c$} &"
                 r" \multicolumn{2}{c}{$c_o$} &"
                 r" \multicolumn{2}{c}{$c_c$} &"
                 r" \multicolumn{2}{c}{FPR$_o$} &"
                 r" \multicolumn{2}{c}{FPR$_c$} \\")
    lines.append(r"  \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
                 r" \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}")
    lines.append(r"  & OpenAI & Qwen & OpenAI & Qwen & OpenAI & Qwen"
                 r" & OpenAI & Qwen & OpenAI & Qwen & OpenAI & Qwen \\")
    lines.append(r"\midrule")

    for ratio in sorted_ratios:
        data = by_ratio[ratio]
        ro = data.get("openai", {}).get("raw", {})
        rc = data.get("openai", {}).get("corr", {})
        qo = data.get("qwen", {}).get("raw", {})
        qc = data.get("qwen", {}).get("corr", {})

        vals = [
            f"{ratio:.2f}",
            fmt_val(ro.get("d")), fmt_val(qo.get("d")),
            fmt_val(rc.get("d")), fmt_val(qc.get("d")),
            fmt_val(ro.get("c")), fmt_val(qo.get("c")),
            fmt_val(rc.get("c")), fmt_val(qc.get("c")),
            fmt_val(ro.get("fpr")), fmt_val(qo.get("fpr")),
            fmt_val(rc.get("fpr")), fmt_val(qc.get("fpr")),
        ]
        line = "  " + " & ".join(vals) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = OUT_DIR / "table_exp1_sdt_corrected.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote LaTeX table to {tex_path}")

    #  per-difficulty raw + corrected 
    diff_results = defaultdict(lambda: defaultdict(dict))

    for (model, ratio), traces in sorted(traces_by_key.items()):
        decided = [r for r in traces if r.get("verifier_decision") is not None and _gold(r) is not None]

        gold_raw_arr = np.array([_gold(r) for r in decided], dtype=bool)
        accept_arr = np.array([r["verifier_decision"] for r in decided], dtype=bool)
        difficulties = np.array([r.get("budget", {}).get("difficulty", "unknown") for r in decided])

        for diff in ("easy", "medium", "hard"):
            mask = difficulties == diff
            if mask.sum() == 0:
                continue
            g = gold_raw_arr[mask]
            a = accept_arr[mask]

            raw_d = _sdt_with_ci(g, a)

            g_corr = g.copy()
            for i, r in enumerate([r for j, r in enumerate(decided) if mask[j]]):
                is_mismatch = r.get("process_outcome_mismatch", False) is True
                verifier_accepted = r["verifier_decision"] is True
                gold_is_false = not bool(g[i])
                if is_mismatch and verifier_accepted and gold_is_false:
                    g_corr[i] = True

            corr_d = _sdt_with_ci(g_corr, a)

            diff_results[(model, ratio)][diff] = {"raw": raw_d, "corr": corr_d}

    diff_name = {"easy": "Easy", "medium": "Med", "hard": "Hard"}

    for metric, met_label in [("d", "$d'$"), ("c", "$c$")]:
        lines = []
        col_spec = r"l*{12}{r}"
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        lines.append(r"\toprule")
        lines.append(r"  \multirow{2}{*}{$R$} &"
                     r" \multicolumn{6}{c}{OpenAI} &"
                     r" \multicolumn{6}{c}{Qwen} \\")
        lines.append(r"  \cmidrule(lr){2-7} \cmidrule(lr){8-13}")
        lines.append(r"  & \multicolumn{2}{c}{Easy} & \multicolumn{2}{c}{Med} & \multicolumn{2}{c}{Hard}"
                     r" & \multicolumn{2}{c}{Easy} & \multicolumn{2}{c}{Med} & \multicolumn{2}{c}{Hard} \\")
        lines.append(r"  \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
                     r" \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}")
        base = met_label.strip("$")
        lines.append(f"  & ${base}_o$ & ${base}_c$"
                     f" & ${base}_o$ & ${base}_c$"
                     f" & ${base}_o$ & ${base}_c$"
                     f" & ${base}_o$ & ${base}_c$"
                     f" & ${base}_o$ & ${base}_c$"
                     f" & ${base}_o$ & ${base}_c$ \\\\")
        lines.append(r"\midrule")

        for ratio in sorted_ratios:
            vals = [f"{ratio:.2f}"]
            for model in ("openai", "qwen"):
                for diff in ("easy", "medium", "hard"):
                    entry = diff_results.get((model, ratio), {}).get(diff, {})
                    raw_v = entry.get("raw", {}).get(metric) if entry else None
                    corr_v = entry.get("corr", {}).get(metric) if entry else None
                    vals.append(fmt_val(raw_v))
                    vals.append(fmt_val(corr_v))
            lines.append("  " + " & ".join(vals) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        fname = f"table_exp1_{metric}_diff_corrected.tex"
        tex_path = OUT_DIR / fname
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote LaTeX table to {tex_path}")


if __name__ == "__main__":
    compute()
