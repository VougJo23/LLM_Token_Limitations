from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import norm

# Recompute SDT / rate metrics with the false-negative correction.

BASE_DIR = Path("./data/experiments/verifier_budget_collapse/main")

FAMILIES = {
    "qwen": {
        "dir": BASE_DIR / "qwen" / "analysis",
        "metrics": ["experiment2_metrics_qwen.json", "experiment2_metrics.json"],
        "labels": ["false_negatives_handlabeled.jsonl",
                   "false_negatives_qwen_handlabeled.jsonl"],
    },
    "openai": {
        "dir": BASE_DIR / "openai" / "analysis",
        "metrics": ["experiment2_metrics_openai.json", "experiment2_metrics.json"],
        "labels": ["false_negatives_openai_handlabeled.jsonl",
                   "false_negatives_handlabeled.jsonl"],
    },
}

OUTPUT_FILE = BASE_DIR / "experiment2_fn_corrected.json"


def find_first(folder: Path, candidates):
    """Return the first existing candidate in folder, else raise a clear error."""
    for name in candidates:
        p = folder / name
        if p.exists():
            return p
    listing = "\n  ".join(sorted(q.name for q in folder.iterdir())) if folder.is_dir() \
        else "(folder does not exist)"
    raise SystemExit(
        f"None of these were found in {folder}:\n  " + "\n  ".join(candidates)
        + f"\nFolder currently contains:\n  {listing}\n"
        "Fix the filename in FAMILIES at the top of the script, or move the file here."
    )


def dprime_criterion(tp, fp, n_signal, n_noise):
    """Macmillan & Creelman log-linear corrected d' and criterion c."""
    if n_signal <= 0 or n_noise <= 0:
        return None, None
    h = (tp + 0.5) / (n_signal + 1.0)
    fa = (fp + 0.5) / (n_noise + 1.0)
    zh, zfa = float(norm.ppf(h)), float(norm.ppf(fa))
    return zh - zfa, -0.5 * (zh + zfa)


def _block(tp, fp, fn, tn):
    ns, nn = tp + fn, fp + tn
    dpr, crit = dprime_criterion(tp, fp, ns, nn)
    r = lambda x: round(x, 3) if isinstance(x, (int, float)) else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "d_prime": r(dpr), "criterion_c": r(crit),
        "fpr": r(fp / nn) if nn else None,
        "fnr": r(fn / ns) if ns else None,
    }


def _ratio_key(rk):
    try:
        return round(float(str(rk).split("=")[-1]), 4)
    except ValueError:
        return None


def label_counts_by_ratio(labels_path: Path):
    by = defaultdict(Counter)
    with open(labels_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            ratio = round(float(o["ratio"]), 4)
            b = o.get("bucket", "")
            key = "flaw" if b.startswith("flaw") else ("truncated" if b == "truncated_artifact" else "actual")
            by[ratio][key] += 1
    return by


def label_counts_by_ratio_and_difficulty(labels_path: Path):

    by = defaultdict(lambda: defaultdict(Counter))
    all_null = True
    with open(labels_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            ratio = round(float(o["ratio"]), 4)
            diff = o.get("difficulty")
            if diff is not None:
                all_null = False
            b = o.get("bucket", "")
            key = "flaw" if b.startswith("flaw") else ("truncated" if b == "truncated_artifact" else "actual")
            by[ratio][diff][key] += 1
    return by, all_null


def _allocate_proportionally(total, weights):
    total_w = sum(weights)
    if total_w == 0:
        return [0] * len(weights)
    raw = [total * w / total_w for w in weights]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    idx = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in range(remainder):
        floors[idx[i]] += 1
    return floors


def process_family(metrics_path: Path, labels_path: Path):
    metrics = json.load(open(metrics_path, encoding="utf-8"))
    labels = label_counts_by_ratio(labels_path)
    labels_by_diff, diff_all_null = label_counts_by_ratio_and_difficulty(labels_path)
    out = {}
    for rk, rv in metrics["per_ratio"].items():
        ratio = _ratio_key(rk)
        c = rv["counts"]
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        lab = labels.get(ratio, Counter())
        flaw, trunc, actual = lab["flaw"], lab["truncated"], lab["actual"]

        by_diff = {}
        bd = rv.get("by_difficulty", {})
        lab_diff = labels_by_diff.get(ratio, {})
        diffs = ["easy", "medium", "hard"]
        diff_fns = [bd.get(d, {}).get("counts", {}).get("fn", 0) for d in diffs]

        if not diff_all_null:
            diff_flaws = [lab_diff.get(d, Counter()).get("flaw", 0) for d in diffs]
        else:
            diff_flaws = _allocate_proportionally(flaw, diff_fns)

        for i, d in enumerate(diffs):
            dc = bd.get(d, {}).get("counts", {})
            d_tp, d_fp, d_fn, d_tn = (dc.get(k, 0) for k in ("tp", "fp", "fn", "tn"))
            d_flaw = diff_flaws[i]
            if d_fn > 0 and d_flaw > d_fn:
                d_flaw = d_fn
            d_orig = _block(d_tp, d_fp, d_fn, d_tn)
            d_corr = _block(d_tp, d_fp, d_fn - d_flaw, d_tn + d_flaw)
            by_diff[d] = {
                "original": d_orig,
                "corrected": d_corr,
                "flaw_moved_fn_to_tn": d_flaw,
            }

        out[f"ratio={ratio:.2f}"] = {
            "original": _block(tp, fp, fn, tn),
            "corrected": _block(tp, fp, fn - flaw, tn + flaw),  # flaw FN -> TN
            "by_difficulty": by_diff,
            "flaw_moved_fn_to_tn": flaw,
            "truncated_held_as_fn": trunc,
            "actual_false_negative": actual,
            "label_rows": flaw + trunc + actual,
            "metrics_fn": fn,
            "labels_match_metrics_fn": (flaw + trunc + actual == fn),
        }
    return out


def main():
    result = {}
    any_inconsistent = False
    for name, cfg in FAMILIES.items():
        folder = Path(cfg["dir"])
        metrics_path = find_first(folder, cfg["metrics"])
        labels_path = find_first(folder, cfg["labels"])
        print(f"=== {name} ===")
        print(f"  metrics: {metrics_path}")
        print(f"  labels : {labels_path}")
        fam = process_family(metrics_path, labels_path)
        result[name] = fam

        print(f"{'ratio':>6} | {'d_o':>5} {'d_c':>5} | {'c_o':>6} {'c_c':>6} | "
              f"{'FPRo':>5} {'FPRc':>5} | {'FNRo':>5} {'FNRc':>5} | moved  ok")
        for rk, b in fam.items():
            o, c = b["original"], b["corrected"]
            ok = "y" if b["labels_match_metrics_fn"] else "MISMATCH"
            any_inconsistent |= not b["labels_match_metrics_fn"]
            f = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else " n/a "
            print(f"{rk.split('=')[1]:>6} | {f(o['d_prime']):>5} {f(c['d_prime']):>5} | "
                  f"{f(o['criterion_c']):>6} {f(c['criterion_c']):>6} | "
                  f"{f(o['fpr']):>5} {f(c['fpr']):>5} | {f(o['fnr']):>5} {f(c['fnr']):>5} | "
                  f"{b['flaw_moved_fn_to_tn']:>5}  {ok}")
        print()

    result["_meta"] = {
        "correction": "false-negative side only: flawed FN -> TN",
        "exact": "corrected FNR is exact",
        "bounds": "corrected d' is an upper bound; corrected criterion is a lower bound on leniency",
        "correction_method": "Macmillan & Creelman log-linear (+0.5/+1)",
    }
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(OUTPUT_FILE, "w", encoding="utf-8"), indent=2)
    print(f"Wrote {OUTPUT_FILE}")
    if any_inconsistent:
        print("WARNING: some ratio's label rows != metrics FN count — check the file pairing.")


if __name__ == "__main__":
    main()
