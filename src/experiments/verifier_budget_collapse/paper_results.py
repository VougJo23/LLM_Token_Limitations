from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path("./data/experiments/verifier_budget_collapse/main")
FAMILY_DIR = {"qwen": BASE_DIR / "qwen" / "analysis",
              "openai": BASE_DIR / "openai" / "analysis"}
METRICS_NAMES = {"qwen": ["experiment2_metrics_qwen.json", "experiment2_metrics.json"],
                 "openai": ["experiment2_metrics_openai.json", "experiment2_metrics.json"]}
FN_CORRECTED = BASE_DIR / "experiment2_fn_corrected.json"
OUT_DIR = BASE_DIR / "paper_results"

FAM_LABEL = {"qwen": "Qwen2.5-7B", "openai": "gpt-4.1-mini"}
ATTACKS = ["arithmetic", "assumption", "mismatch", "persuasive"]
DIFFS = ["easy", "medium", "hard"]
RATIOS = [0.30, 0.45, 0.60, 0.75, 0.90]


def find_first(folder, names):
    for n in names:
        p = folder / n
        if p.exists():
            return p
    raise SystemExit(f"None of {names} found in {folder}")

def load():
    metrics = {f: json.load(open(find_first(FAMILY_DIR[f], METRICS_NAMES[f]))) for f in FAMILY_DIR}
    fnc = json.load(open(FN_CORRECTED))
    return metrics, fnc

def per_ratio_sorted(metrics_fam):
    return [metrics_fam["per_ratio"][k] for k in sorted(metrics_fam["per_ratio"],
            key=lambda x: float(x.split("=")[1]))]

def attack_fpr_mean(metrics_fam):
    out = {}
    for a in ATTACKS:
        vals = [rv["by_attack"][a]["rates"]["false_positive_rate"] for rv in metrics_fam["per_ratio"].values()]
        out[a] = sum(vals) / len(vals)
    return out

def diff_dprime(fnc_fam):
    out = {}
    ratios = sorted(fnc_fam, key=lambda x: float(x.split("=")[1]))
    for diff in DIFFS:
        dp = []
        for rk in ratios:
            bd = fnc_fam[rk].get("by_difficulty", {}).get(diff, {})
            d = bd.get("corrected", {}).get("d_prime")
            if d is not None:
                dp.append(d)
        out[diff] = {
            "d_mean": sum(dp) / len(dp) if dp else 0.0,
            "d_min": min(dp) if dp else 0.0,
            "d_max": max(dp) if dp else 0.0,
        }
    return out


def fnc_rows(fnc, fam):
    return [fnc[fam][f"ratio={r:.2f}"] for r in RATIOS]


def overview_rows(metrics_fam, fnc, fam):
    rows = []
    for r in RATIOS:
        rk = f"ratio={r:.2f}"
        m = metrics_fam["per_ratio"][rk]
        corr = fnc[fam][rk]["corrected"]
        cts = m["counts"]
        gen_correct_corr = corr["tp"] + corr["fn"] + cts["abstain_correct"]
        rows.append({
            "ratio": r,
            "gen_acc_corr": gen_correct_corr / cts["n_valid"],
            "ver_acc": m["accuracy"]["system_accuracy_valid"],
            "fnr_c": corr["fnr"],
            "fpr": m["rates"]["false_positive_rate"],
            #"mfpr": metrics_fam["headline_findings"]["mismatch_fpr_by_ratio"][rk],
            "edr": m["rates"]["error_detection_rate"],
            "dprime_c": corr["d_prime"],
            "crit_c": corr["criterion_c"],
            #"contra": m["step_contradiction"]["contradiction_rate"],
            
        })
    return rows


# LaTeX tables
def tex_overview(metrics, fnc):
    q_rows = overview_rows(metrics["qwen"], fnc, "qwen")
    o_rows = overview_rows(metrics["openai"], fnc, "openai")
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Experiment~2 overview. Per verifier-budget ratio and verifier family: "
        r"corrected generator accuracy (after FN relabel), verifier accuracy, false-negative rate "
        r"(FNR$_c$, corrected), false-positive rate (FPR, wrong answer accepted), "
        r"error-detection rate (EDR), and sensitivity $d'_c$ and criterion $c_c$ (corrected, "
        r"log-linear). $d'_c$/FNR$_c$/$c_c$/gen acc use the false-negative label correction; "
        r"$d'_c$ is an upper bound and $c_c$ a lower bound on leniency.}",
        r"\label{tab:overview}",
        r"\footnotesize",
        r"\begin{tabular}{l*{14}{r}}", r"\toprule",
        r"ratio & \multicolumn{2}{c}{gen acc} & \multicolumn{2}{c}{ver acc} & \multicolumn{2}{c}{FNR$_c$} & \multicolumn{2}{c}{FPR} & \multicolumn{2}{c}{EDR} & \multicolumn{2}{c}{$d'_c$} & \multicolumn{2}{c}{$c_c$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13} \cmidrule(lr){14-15}",
        r" & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} \\",
        r"\midrule",
    ]
    for r_idx, r in enumerate(RATIOS):
        q, o = q_rows[r_idx], o_rows[r_idx]
        lines.append(
            f"{r:.2f} & "
            f"{q['gen_acc_corr']:.3f} & {o['gen_acc_corr']:.3f} & "
            f"{q['ver_acc']:.3f} & {o['ver_acc']:.3f} & "
            f"{q['fnr_c']:.3f} & {o['fnr_c']:.3f} & "
            f"{q['fpr']:.3f} & {o['fpr']:.3f} & "
            f"{q['edr']:.3f} & {o['edr']:.3f} & "
            f"{q['dprime_c']:.2f} & {o['dprime_c']:.2f} & "
            f"{q['crit_c']:.2f} & {o['crit_c']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def tex_table1(fnc):
    q_rows = fnc_rows(fnc, "qwen")
    o_rows = fnc_rows(fnc, "openai")
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Signal-detection metrics per verifier budget ratio, as collected "
        r"versus after the false-negative label correction (flawed-but-correct-answer "
        r"traces reassigned from the signal to the noise class). $d'$ and criterion $c$ "
        r"use the Macmillan--Creelman log-linear correction. FNR$_c$ is exact; $d'_c$ is "
        r"an upper bound and $c_c$ a lower bound on leniency (accept-side audit not applied).}",
        r"\label{tab:sdt_corrected}",
        r"\small",
        r"\begin{tabular}{l*{14}{r}}", r"\toprule",
        r"ratio & \multicolumn{2}{c}{$d'_o$} & \multicolumn{2}{c}{$d'_c$} & \multicolumn{2}{c}{$c_o$} & \multicolumn{2}{c}{$c_c$} & \multicolumn{2}{c}{FPR} & \multicolumn{2}{c}{FNR$_o$} & \multicolumn{2}{c}{FNR$_c$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13} \cmidrule(lr){14-15}",
        r" & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} \\",
        r"\midrule",
    ]
    for r_idx, r in enumerate(RATIOS):
        qo, qc = q_rows[r_idx]["original"], q_rows[r_idx]["corrected"]
        oo, oc = o_rows[r_idx]["original"], o_rows[r_idx]["corrected"]
        lines.append(
            f"{r:.2f} & {qo['d_prime']:.2f} & {oo['d_prime']:.2f} & "
            f"{qc['d_prime']:.2f} & {oc['d_prime']:.2f} & "
            f"{qo['criterion_c']:.2f} & {oo['criterion_c']:.2f} & "
            f"{qc['criterion_c']:.2f} & {oc['criterion_c']:.2f} & "
            f"{qo['fpr']:.3f} & {oo['fpr']:.3f} & "
            f"{qo['fnr']:.3f} & {oo['fnr']:.3f} & "
            f"{qc['fnr']:.3f} & {oc['fnr']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def tex_table2(metrics):
    qa = attack_fpr_mean(metrics["qwen"])
    oa = attack_fpr_mean(metrics["openai"])
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Mean false-positive rate (wrong answer accepted) by attack type, "
        r"averaged over budget ratios. The mismatch column shows the cross-family "
        r"dissociation: near-total for Qwen, near-zero for gpt-4.1-mini.}",
        r"\label{tab:fpr_attack}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"attack & " + " & ".join(FAM_LABEL[f] for f in ("qwen", "openai")) + r" \\",
        r"\midrule",
    ]
    for a in ATTACKS:
        lines.append(f"{a} & {qa[a]:.3f} & {oa[a]:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tex_table3(fnc):
    qd = diff_dprime(fnc["qwen"])
    od = diff_dprime(fnc["openai"])
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Mean corrected sensitivity $d'_c$ by problem difficulty (averaged "
        r"over budget ratios), with range across ratios in brackets. Sensitivity is "
        r"difficulty-gated and roughly budget-invariant within each level "
        r"(the bracketed ranges are narrow).}",
        r"\label{tab:dprime_difficulty}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"difficulty & " + " & ".join(FAM_LABEL[f] for f in ("qwen", "openai")) + r" \\",
        r"\midrule",
    ]
    for diff in DIFFS:
        q, o = qd[diff], od[diff]
        lines.append(f"{diff} & {q['d_mean']:.2f} [{q['d_min']:.2f},{q['d_max']:.2f}] & "
                     f"{o['d_mean']:.2f} [{o['d_min']:.2f},{o['d_max']:.2f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tex_table4(fnc):
    q_rows = fnc_rows(fnc, "qwen")
    o_rows = fnc_rows(fnc, "openai")
    q_nom = sum(r["metrics_fn"] for r in q_rows)
    o_nom = sum(r["metrics_fn"] for r in o_rows)
    q_tr  = sum(r["truncated_held_as_fn"] for r in q_rows)
    o_tr  = sum(r["truncated_held_as_fn"] for r in o_rows)
    q_fl  = sum(r["flaw_moved_fn_to_tn"] for r in q_rows)
    o_fl  = sum(r["flaw_moved_fn_to_tn"] for r in o_rows)
    q_ac  = sum(r["actual_false_negative"] for r in q_rows)
    o_ac  = sum(r["actual_false_negative"] for r in o_rows)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Decomposition of nominal false negatives (gold-correct, verifier-rejected) "
        r"summed over budget ratios. Only the \emph{actual} column are true verifier errors; "
        r"truncated verdicts and correctly-rejected flawed traces inflate the raw count.}",
        r"\label{tab:fn_correction}",
        r"\begin{tabular}{l*{8}{r}}", r"\toprule",
        r" & \multicolumn{2}{c}{nominal FN} & \multicolumn{2}{c}{truncated} & \multicolumn{2}{c}{flaw} & \multicolumn{2}{c}{actual FN} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r" & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} & \textsc{qwen} & \textsc{openai} \\",
        r"\midrule",
        f"total & {q_nom} & {o_nom} & {q_tr} & {o_tr} & {q_fl} & {o_fl} & {q_ac} & {o_ac} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# Plots
def setup_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })

COL = {"qwen": "#c1432f", "openai": "#2f5fc1"}

def save(fig, stem):
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png")
    plt.close(fig)

def fig1_dprime_criterion(fnc):
    fig, (axd, axc) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for fam in ("qwen", "openai"):
        rows = fnc_rows(fnc, fam)
        d_o = [r["original"]["d_prime"] for r in rows]
        d_c = [r["corrected"]["d_prime"] for r in rows]
        c_o = [r["original"]["criterion_c"] for r in rows]
        c_c = [r["corrected"]["criterion_c"] for r in rows]
        axd.plot(RATIOS, d_o, "--o", color=COL[fam], alpha=0.45, mfc="white", label=f"{FAM_LABEL[fam]} (orig)")
        axd.plot(RATIOS, d_c, "-o", color=COL[fam], label=f"{FAM_LABEL[fam]} (corr)")
        axc.plot(RATIOS, c_o, "--o", color=COL[fam], alpha=0.45, mfc="white")
        axc.plot(RATIOS, c_c, "-o", color=COL[fam])
    axd.set_title(r"Sensitivity $d'$ (flat across budget)")
    axd.set_xlabel("verifier budget ratio"); axd.set_ylabel(r"$d'$"); axd.set_ylim(0)
    axd.legend(fontsize=7, frameon=False)
    axc.axhline(0, color="0.6", lw=0.8, ls=":")
    axc.set_title(r"Criterion $c$ (lenient, shifts; never $>0$)")
    axc.set_xlabel("verifier budget ratio"); axc.set_ylabel(r"$c$")
    save(fig, "fig1_dprime_criterion")


def fig2_fpr_by_attack(metrics):
    qa = attack_fpr_mean(metrics["qwen"]); oa = attack_fpr_mean(metrics["openai"])
    import numpy as np
    x = np.arange(len(ATTACKS)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(x - w / 2, [qa[a] for a in ATTACKS], w, color=COL["qwen"], label=FAM_LABEL["qwen"])
    ax.bar(x + w / 2, [oa[a] for a in ATTACKS], w, color=COL["openai"], label=FAM_LABEL["openai"])
    ax.set_xticks(x); ax.set_xticklabels(ATTACKS)
    ax.set_ylabel("mean FPR (wrong accepted)")
    ax.set_title("Attack vulnerability by verifier family")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig2_fpr_by_attack")

def fig3_dprime_by_difficulty(fnc):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharex=True)
    dcol = {"easy": "#2f8f4e", "medium": "#c98a2f", "hard": "#b03a3a"}
    for ax, fam in zip(axes, ("qwen", "openai")):
        rows = fnc_rows(fnc, fam)
        for diff in DIFFS:
            dp = [r["by_difficulty"][diff]["corrected"]["d_prime"] for r in rows]
            ax.plot(RATIOS, dp, "-o", color=dcol[diff], label=diff)
        ax.set_title(FAM_LABEL[fam]); ax.set_xlabel("verifier budget ratio"); ax.set_ylim(0)
    axes[0].set_ylabel(r"$d'_c$"); axes[0].legend(frameon=False, fontsize=8, title="difficulty")
    fig.suptitle("Difficulty-gated sensitivity (level effect, budget-invariant)", y=1.02)
    save(fig, "fig3_dprime_by_difficulty")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, fnc = load()
    setup_style()

    tables = {
        "table0_overview.tex": tex_overview(metrics, fnc),
        "table1_sdt_corrected.tex": tex_table1(fnc),
        "table2_fpr_by_attack.tex": tex_table2(metrics),
        "table3_dprime_by_difficulty.tex": tex_table3(fnc),
        "table4_fn_correction.tex": tex_table4(fnc),
    }
    for name, body in tables.items():
        (OUT_DIR / name).write_text(body + "\n", encoding="utf-8")
    (OUT_DIR / "all_tables.tex").write_text(
        "\n\n".join(r"\input{%s}" % n.replace(".tex", "") for n in tables) + "\n", encoding="utf-8")

    fig1_dprime_criterion(fnc)
    fig2_fpr_by_attack(metrics)
    fig3_dprime_by_difficulty(fnc)

    print(f"Wrote {len(tables)} tables and 3 figures (pdf+png) to {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
