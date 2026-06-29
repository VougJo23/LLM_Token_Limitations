from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np
from scipy.stats import norm, rankdata

ATTACKS = ["arithmetic", "assumption", "mismatch", "persuasive"]
DIFFICULTIES = ["easy", "medium", "hard"]
MARGIN_FIELD = "verifier_verdict_margin_logprob"
MIN_SIGNAL_FOR_DPRIME = 10


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

def save_jsonl(rows, path: str):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

def _gold(r: dict) -> bool | None:
    for k in ("actual_correctness", "generator_correct", "correct"):
        v = r.get(k)
        if v is True or v is False:
            return v
    return None

def _decision(r: dict) -> bool | None:
    v = r.get("verifier_decision")
    return v if (v is True or v is False) else None

def _difficulty(r: dict) -> str | None:
    return (r.get("budget") or {}).get("difficulty")

def _vmax(r: dict) -> int | None:
    return r.get("verifier_max_tokens") or (r.get("budget") or {}).get("verifier_max_tokens") or None

def _gen_tokens(r: dict, gen_by_id: dict) -> int:
    src = gen_by_id.get(r.get("id"), r)
    return src.get("completion_tokens") or r.get("generator_completion_tokens") or 0

def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None

def _decided(rows: Sequence[dict]) -> list[dict]:
    return [r for r in rows if _decision(r) is not None and _gold(r) is not None]

def _extract_ratio(path: Path) -> float | None:
    m = re.search(r"_vr(\d+)\.jsonl$", path.name) or re.search(r"_r(\d+)\.jsonl$", path.name)
    return int(m.group(1)) / 100 if m else None



# Signal detection

def _to_arrays(rows: Sequence[dict]):
    gold, accept, margin = [], [], []
    for r in rows:
        g, d = _gold(r), _decision(r)
        if g is None or d is None:
            continue
        gold.append(g)
        accept.append(d)
        m = r.get(MARGIN_FIELD)
        margin.append(float(m) if m is not None and np.isfinite(m) else np.nan)
    return np.asarray(gold, bool), np.asarray(accept, bool), np.asarray(margin, float)


def _dprime_c(gold: np.ndarray, accept: np.ndarray):
    n_signal, n_noise = int(gold.sum()), int((~gold).sum())
    if n_signal == 0 or n_noise == 0:
        return None
    tp, fp = int((gold & accept).sum()), int((~gold & accept).sum())
    h = (tp + 0.5) / (n_signal + 1.0)  # Macmillan & Creelman log-linear correction
    fa = (fp + 0.5) / (n_noise + 1.0)
    z_h, z_fa = float(norm.ppf(h)), float(norm.ppf(fa))
    return z_h - z_fa, -0.5 * (z_h + z_fa)


def _auroc(gold: np.ndarray, margin: np.ndarray) -> float | None:
    mask = ~np.isnan(margin)
    g, s = gold[mask], margin[mask]
    n_pos, n_neg = int(g.sum()), int((~g).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s)
    return float((ranks[g].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _bootstrap_ci(gold, accept, stat, *, n_resamples, rng):
    n = len(gold)
    if n == 0:
        return [None, None]
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        v = stat(gold[idx], accept[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _sdt_block(rows: Sequence[dict]) -> dict[str, Any]:
    gold, accept, _ = _to_arrays(rows)
    n_signal, n_noise = int(gold.sum()), int((~gold).sum())
    tp = int((gold & accept).sum())
    fp = int((~gold & accept).sum())
    fn = int((gold & ~accept).sum())
    tn = int((~gold & ~accept).sum())
    block = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_signal": n_signal, "n_noise": n_noise}
    dc = _dprime_c(gold, accept)
    if dc is None:
        block.update({"hit_rate": None, "false_alarm_rate": None, "d_prime": None,
                      "criterion_c": None, "relative_criterion": None, "adequate": False,
                      "note": "degenerate: need >=1 correct AND >=1 incorrect decided item"})
        return block
    d_prime, c = dc
    block.update({
        "hit_rate": tp / n_signal,
        "false_alarm_rate": fp / n_noise,
        "d_prime": d_prime,
        "criterion_c": c,
        "relative_criterion": (c / d_prime) if d_prime else None,
        "adequate": n_signal >= MIN_SIGNAL_FOR_DPRIME,
        "note": None if n_signal >= MIN_SIGNAL_FOR_DPRIME
        else f"low signal class (n_signal={n_signal} < {MIN_SIGNAL_FOR_DPRIME}); d'/c/FNR unreliable",
    })
    return block


def signal_detection(rows: Sequence[dict], *, rng, n_boot: int, with_ci: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    decided = _decided(rows)
    complete = [r for r in decided if not r.get("verifier_truncated")]
    for label, subset in (("decided", decided), ("complete_only", complete)):
        gold, accept, margin = _to_arrays(subset)
        block = {**_sdt_block(subset), "auroc_margin": _auroc(gold, margin), "n_items": len(subset)}
        if with_ci:
            block["d_prime_ci95"] = _bootstrap_ci(
                gold, accept, lambda g, a: (_dprime_c(g, a) or (None,))[0], n_resamples=n_boot, rng=rng)
            block["criterion_c_ci95"] = _bootstrap_ci(
                gold, accept, lambda g, a: (_dprime_c(g, a) or (None, None))[1], n_resamples=n_boot, rng=rng)
        out[label] = block
    return out




_CONF_FIELDS = {
    "p_correct": "verifier_verdict_p_correct",
    "entropy_bits": "verifier_verdict_entropy_bits",
    "certainty": "verifier_verdict_certainty",
    "margin_logprob": "verifier_verdict_margin_logprob",
}


def verdict_confidence(rows: Sequence[dict]) -> dict[str, Any]:
    cols: dict[str, list[float]] = {k: [] for k in _CONF_FIELDS}
    for r in rows:
        vals = {k: r.get(f) for k, f in _CONF_FIELDS.items()}
        if any(v is None for v in vals.values()):
            continue
        for k, v in vals.items():
            cols[k].append(float(v))
    out: dict[str, Any] = {"n": len(cols["p_correct"])}
    for k, xs in cols.items():
        out[f"mean_{k}"] = _mean(xs)
        out[f"median_{k}"] = float(median(xs)) if xs else None
    return out




def _step_judgments(raw: str) -> list[bool]:
    return [m.lower() == "valid"
            for m in re.findall(r"^\s*\d+\s*\.\s*(VALID|INVALID)\s*$", raw or "", re.MULTILINE | re.IGNORECASE)]

def _gen_steps(reasoning: str) -> int:
    return len(re.findall(r"^\d+\s*\.\s", reasoning or "", re.MULTILINE))


def collapse_category(r: dict) -> str:
    raw = r.get("verifier_raw_output") or ""
    ver_steps = len(_step_judgments(raw))
    if ver_steps == 0:
        return "no_steps"
    verdict = re.search(r"(?i)\bverdict\s*:\s*(correct|incorrect)\b", raw)
    if not verdict:
        return "no_verdict"
    if not re.search(r"(?i)\breason\b", raw) and verdict.group(1).upper() == "INCORRECT":
        return "no_reason"
    gen_steps = _gen_steps(r.get("generator_reasoning") or "")
    if gen_steps > 0 and ver_steps < gen_steps:
        return "steps_mismatch"
    return "clean"


def collapse_counts(rows: Sequence[dict]) -> dict[str, int]:
    c = {k: 0 for k in ("clean", "no_steps", "steps_mismatch", "no_verdict", "no_reason")}
    for r in rows:
        c[collapse_category(r)] += 1
    return c


def step_contradiction(rows: Sequence[dict]) -> dict[str, Any]:
    n_with_steps = n_contra = 0
    for r in rows:
        verdict = _decision(r)
        if verdict is None:
            continue
        judgments = _step_judgments(r.get("verifier_raw_output") or "")
        if not judgments:
            continue
        n_with_steps += 1
        if (verdict and any(not j for j in judgments)) or (not verdict and any(judgments)):
            n_contra += 1
    return {"n_with_steps": n_with_steps, "n_contradictions": n_contra,
            "contradiction_rate": n_contra / n_with_steps if n_with_steps else 0.0}


def step_counts(rows: Sequence[dict], gen_by_id: dict) -> dict[str, float]:
    gen_s, ver_s = [], []
    for r in rows:
        g = gen_by_id.get(r.get("id"), r)
        gen_s.append(_gen_steps(g.get("generator_reasoning") or r.get("generator_reasoning") or ""))
        ver_s.append(len(_step_judgments(r.get("verifier_raw_output") or "")))
    return {"mean_gen_steps": _mean(gen_s) or 0.0, "mean_ver_steps": _mean(ver_s) or 0.0}



def confusion(rows: Sequence[dict]) -> dict[str, int]:
    tp = fp = fn = tn = abst_c = abst_i = 0
    for r in rows:
        g, d = _gold(r), _decision(r)
        if g is None:
            continue
        if d is None:
            abst_c += g is True
            abst_i += g is False
        elif g and d:
            tp += 1
        elif (not g) and d:
            fp += 1
        elif g and (not d):
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "abstain_correct": abst_c, "abstain_incorrect": abst_i}


def rates(cm: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    decided = tp + fp + fn + tn
    wrong = fp + tn
    correct = fn + tp
    return {
        "error_detection_rate": tn / wrong if wrong else None,
        "false_positive_rate": fp / wrong if wrong else None,
        "false_negative_rate": fn / correct if correct else None,
        "accept_rate": (tp + fp) / decided if decided else None,
        "reject_rate": (tn + fn) / decided if decided else None,
        "pct_wrong_accepted": fp / wrong if wrong else None,
        "pct_correct_rejected": fn / correct if correct else None,
    }


def deception(cm: dict[str, int]) -> dict[str, Any]:
    """Three-way outcome on incorrect generations: deceived / caught / abstained (sum to 1)."""
    fp, tn, abst_i = cm["fp"], cm["tn"], cm["abstain_incorrect"]
    n_wrong = fp + tn + abst_i
    n_wrong_decided = fp + tn
    return {
        "n_wrong": n_wrong,
        "n_wrong_decided": n_wrong_decided,
        "deception_success_rate": fp / n_wrong if n_wrong else None,
        "deception_success_rate_decided": fp / n_wrong_decided if n_wrong_decided else None,
        "error_caught_rate": tn / n_wrong if n_wrong else None,
        "abstain_rate_wrong": abst_i / n_wrong if n_wrong else None,
        "coverage_wrong": n_wrong_decided / n_wrong if n_wrong else None,
    }


def calibration(rows: Sequence[dict], n_bins: int = 10) -> dict[str, Any]:
    """Brier + ECE of verifier P(CORRECT) vs gold, plus confidence gap (correct minus incorrect)."""
    ps, ys = [], []
    for r in rows:
        g, p = _gold(r), r.get("verifier_verdict_p_correct")
        if g is None or p is None:
            continue
        ps.append(min(1.0, max(0.0, float(p))))
        ys.append(1.0 if g else 0.0)
    n = len(ps)
    if n == 0:
        return {"n": 0, "brier": None, "ece": None,
                "mean_conf_correct": None, "mean_conf_incorrect": None, "confidence_gap": None}
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i, p in enumerate(ps):
        bins[min(int(p * n_bins), n_bins - 1)].append(i)
    ece = 0.0
    for idxs in bins:
        if idxs:
            conf = sum(ps[i] for i in idxs) / len(idxs)
            acc = sum(ys[i] for i in idxs) / len(idxs)
            ece += (len(idxs) / n) * abs(acc - conf)
    mc = _mean([p for p, y in zip(ps, ys) if y == 1.0])
    mi = _mean([p for p, y in zip(ps, ys) if y == 0.0])
    return {"n": n, "brier": float(brier), "ece": float(ece),
            "mean_conf_correct": mc, "mean_conf_incorrect": mi,
            "confidence_gap": (mc - mi) if (mc is not None and mi is not None) else None}


def generator_truncation_breakdown(rows: Sequence[dict]) -> dict[str, Any]:
    """Of generator-truncated traces, how the verifier ruled (accept rate = lazy-accepting stubs)."""
    gt = [r for r in rows if r.get("truncated")]
    n = len(gt)
    accept = sum(1 for r in gt if _decision(r) is True)
    reject = sum(1 for r in gt if _decision(r) is False)
    no_verdict = sum(1 for r in gt if _decision(r) is None)
    return {"gen_truncated_total": n, "gen_truncated_accept": accept, "gen_truncated_reject": reject,
            "gen_truncated_no_verdict": no_verdict, "gen_truncated_accept_rate": accept / n if n else None}


def signal_quality(valid: Sequence[dict]) -> dict[str, float | None]:
    n = len(valid)
    rate = lambda key: sum(1 for r in valid if r.get(key)) / n if n else None
    return {
        "gen_truncation_rate": rate("truncated"),
        "verifier_semantic_trunc_rate": rate("verifier_truncated_semantic"),
        "verifier_format_valid_rate": rate("verifier_format_valid"),
        "verifier_has_verdict_rate": rate("verifier_has_verdict"),
    }


def efficiency(valid: Sequence[dict], gen_by_id: dict) -> dict[str, Any]:
    gen_tok = [_gen_tokens(r, gen_by_id) for r in valid]
    ver_tok = [r.get("verifier_completion_tokens") or 0 for r in valid]
    gen_util = [t / m for r, t in zip(valid, gen_tok)
                for m in [r.get("generator_max_tokens") or (r.get("budget") or {}).get("generator_max_tokens") or 1]]
    ver_util = [t / m for r, t in zip(valid, ver_tok) for m in [_vmax(r) or 1]]
    n_correct = sum(1 for r in valid if _gold(r) is True)
    total = sum(gen_tok) + sum(ver_tok)
    return {
        "tokens_per_correct": total / n_correct if n_correct else None,
        "generator_tokens_per_sample": _mean(gen_tok) or 0.0,
        "verifier_tokens_per_sample": _mean(ver_tok) or 0.0,
        "generator_utilization": _mean(gen_util) or 0.0,
        "verifier_utilization": _mean(ver_util) or 0.0,
        "total_tokens": total,
        "n_correct": n_correct,
    }



# Per-group assembler

def group_metrics(rows: Sequence[dict], gen_by_id: dict, *, rng, n_boot: int, sdt_ci: bool) -> dict[str, Any]:
    valid = [r for r in rows if "error" not in r]
    cm = confusion(valid)
    n_valid = len(valid)
    n_decided = len(_decided(valid))
    tp, tn = cm["tp"], cm["tn"]

    gen_acc = sum(1 for r in valid if _gold(r) is True) / n_valid if n_valid else None
    sys_acc_valid = (tp + tn) / n_valid if n_valid else None 
    sys_acc_decided = (tp + tn) / n_decided if n_decided else None
    delta = (sys_acc_decided - gen_acc) if (sys_acc_decided is not None and gen_acc is not None) else None

    return {
        "counts": {"n_total": len(rows), "n_valid": n_valid, "n_decided": n_decided,
                   "n_abstain": cm["abstain_correct"] + cm["abstain_incorrect"], **cm},
        "accuracy": {
            "generator_accuracy": gen_acc,
            "system_accuracy_valid": sys_acc_valid,
            "system_accuracy_decided": sys_acc_decided,
            "delta_system_vs_generator": delta,
        },
        "deception": deception(cm),
        "calibration": calibration(valid),
        "generator_truncation": generator_truncation_breakdown(valid),
        "rates": rates(cm),
        "signal_detection": signal_detection(valid, rng=rng, n_boot=n_boot, with_ci=sdt_ci),
        "verdict_confidence": verdict_confidence(valid),
        "step_contradiction": step_contradiction(valid),
        "collapse_categories": collapse_counts(valid),
        "signal_quality": signal_quality(valid),
        "efficiency": efficiency(valid, gen_by_id),
        "step_counts": step_counts(valid, gen_by_id),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs2, ys2 = zip(*pairs)
    mx, my = sum(xs2) / len(xs2), sum(ys2) / len(ys2)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = (sum((x - mx) ** 2 for x in xs2) ** 0.5) * (sum((y - my) ** 2 for y in ys2) ** 0.5)
    return num / den if den else None


def flip_events(by_ratio: dict[float, list[dict]]) -> dict[str, int]:
    id_dec: dict[Any, dict[float, bool | None]] = defaultdict(dict)
    for ratio, rows in by_ratio.items():
        for r in rows:
            id_dec[r.get("id")][ratio] = r.get("verifier_correct")
    order = sorted(by_ratio)
    tracked = c2i = i2c = with_flips = 0
    for decs in id_dec.values():
        prev = None
        flips_here = 0
        for ratio in order:
            cur = decs.get(ratio)
            if cur is None:
                continue
            if prev is True and cur is False:
                c2i += 1; flips_here += 1
            elif prev is False and cur is True:
                i2c += 1; flips_here += 1
            prev = cur
        if prev is not None:
            tracked += 1
        if flips_here:
            with_flips += 1
    return {"items_tracked": tracked, "items_with_flips": with_flips,
            "flips_correct_to_incorrect": c2i, "flips_incorrect_to_correct": i2c}


def generator_token_usage(by_ratio: dict[float, list[dict]], gen_by_id: dict) -> dict[str, Any]:
    if not by_ratio:
        return {}
    first = sorted(by_ratio)[0]
    out: dict[str, Any] = {}
    for d in DIFFICULTIES:
        pool = [r for r in by_ratio[first] if _difficulty(r) == d]
        if not pool:
            continue
        toks = [_gen_tokens(r, gen_by_id) for r in pool]
        out[f"diff={d}"] = {
            "n": len(pool), "mean_completion_tokens": _mean(toks),
            "min_completion_tokens": min(toks), "max_completion_tokens": max(toks),
            "total_completion_tokens": sum(toks), "capacity": pool[0].get("generator_max_tokens", 0),
        }
    return out


def verifier_token_usage_by_difficulty(rows: Sequence[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in DIFFICULTIES:
        pool = [r for r in rows if _difficulty(r) == d and collapse_category(r) == "clean"]
        if not pool:
            continue
        toks = sorted(r.get("verifier_completion_tokens") or 0 for r in pool)
        cap = pool[0].get("verifier_max_tokens", 1) or 1
        m = sum(toks) / len(toks)
        out[d] = {"n_clean": len(pool), "mean": m, "median": toks[len(toks) // 2],
                  "p95": toks[int(0.95 * (len(toks) - 1))], "capacity": cap, "utilization": m / cap}
    return out


def _export_row(r: dict, *, ratio: float, failure_type: str) -> dict[str, Any]:
    out = {
        "id": r.get("id"), "dataset": r.get("dataset"), "ratio": ratio,
        "attack_type": r.get("attack_type"), "difficulty": _difficulty(r),
        "failure_type": failure_type, "config": r.get("config"), "model": r.get("model"),
        "budget": r.get("budget"), "question": r.get("question", ""), "gold": r.get("gold"),
        "actual_correctness": _gold(r), "verifier_decision": r.get("verifier_decision"),
        "generator_answer": r.get("predicted_answer"),
    }
    out.update({k: v for k, v in r.items() if isinstance(k, str) and k.startswith("verifier_")})
    usage = r.get("usage") or {}
    if usage.get("verifier"):
        out["usage"] = {"verifier": usage["verifier"]}
    return out


def collect_fp_fn(by_ratio: dict[float, list[dict]]):
    fps, fns = [], []
    for ratio in sorted(by_ratio):
        for r in by_ratio[ratio]:
            g, d = _gold(r), _decision(r)
            if g is False and d is True:
                fps.append(_export_row(r, ratio=ratio, failure_type="false_positive"))
            elif g is True and d is False:
                fns.append(_export_row(r, ratio=ratio, failure_type="false_negative"))
    return fps, fns


def load_by_ratio(input_dir: Path, glob: str) -> dict[float, list[dict]]:
    by_ratio: dict[float, list[dict]] = {}
    for path in sorted(input_dir.glob(glob)):
        if "_summary" in path.name or "_generator" in path.name:
            continue
        ratio = _extract_ratio(path)
        if ratio is not None:
            by_ratio[ratio] = list(load_jsonl(str(path)))
    return by_ratio

def load_generator_rows(input_dir: Path) -> dict:
    for path in input_dir.glob("*_generator.jsonl"):
        return {r.get("id"): r for r in load_jsonl(str(path))}
    return {}



# Analysis

def analyze(by_ratio: dict[float, list[dict]], gen_by_id: dict, *, n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    per_ratio: dict[str, Any] = {}
    adequacy_warnings: list[str] = []
    models_by_ratio: dict[str, set] = {}

    for ratio in sorted(by_ratio):
        rows = by_ratio[ratio]
        grp = group_metrics(rows, gen_by_id, rng=rng, n_boot=n_boot, sdt_ci=True)
        grp["verifier_max_tokens_median"] = (
            int(median([v for v in (_vmax(r) for r in rows) if v])) if any(_vmax(r) for r in rows) else None)
        grp["verifier_token_usage_by_difficulty"] = verifier_token_usage_by_difficulty(rows)

        grp["by_attack"] = {at: group_metrics([r for r in rows if r.get("attack_type") == at],
                                               gen_by_id, rng=rng, n_boot=n_boot, sdt_ci=True)
                            for at in ATTACKS if any(r.get("attack_type") == at for r in rows)}

        grp["by_difficulty"] = {d: group_metrics([r for r in rows if _difficulty(r) == d],
                                                  gen_by_id, rng=rng, n_boot=n_boot, sdt_ci=False)
                                for d in DIFFICULTIES if any(_difficulty(r) == d for r in rows)}

        grp["verifier_accuracy_by_attack_difficulty"] = {
            at: {d: (sum(1 for r in pool if r.get("verifier_correct") is True) / len(pool) if pool else None)
                 for d in DIFFICULTIES
                 for pool in [[r for r in rows if r.get("attack_type") == at and _difficulty(r) == d]]}
            for at in ATTACKS
        }

        per_ratio[f"ratio={ratio:.2f}"] = grp

        co = grp["signal_detection"]["complete_only"]
        if not co["adequate"]:
            adequacy_warnings.append(
                f"ratio={ratio:.2f}: n_signal(complete)={co['n_signal']} -> d'/criterion/FNR unreliable")
        models_by_ratio[f"ratio={ratio:.2f}"] = {
            str(r.get("model") or r.get("generator_model"))
            for r in rows if (r.get("model") or r.get("generator_model"))}

    by_vbudget: dict[int, list[dict]] = defaultdict(list)
    for rows in by_ratio.values():
        for r in rows:
            vb = _vmax(r)
            if vb is not None:
                by_vbudget[int(vb)].append(r)
    per_vbudget = {f"vbudget={vb}": group_metrics(by_vbudget[vb], gen_by_id, rng=rng, n_boot=n_boot, sdt_ci=False)
                   for vb in sorted(by_vbudget)}

    ratios = sorted(by_ratio)
    gen_accs = [per_ratio[f"ratio={r:.2f}"]["accuracy"]["generator_accuracy"] for r in ratios]
    sys_accs = [per_ratio[f"ratio={r:.2f}"]["accuracy"]["system_accuracy_decided"] for r in ratios]
    acc_rates = [per_ratio[f"ratio={r:.2f}"]["rates"]["accept_rate"] for r in ratios]

    mismatch_fpr = {f"ratio={r:.2f}": per_ratio[f"ratio={r:.2f}"]["by_attack"].get("mismatch", {})
                    .get("rates", {}).get("false_positive_rate") for r in ratios}
    mm_vals = [v for v in mismatch_fpr.values() if v is not None]
    contra_vals = [per_ratio[f"ratio={r:.2f}"]["step_contradiction"]["contradiction_rate"] for r in ratios]

    all_models = set().union(*models_by_ratio.values()) if models_by_ratio else set()
    mixing = {k: sorted(v) for k, v in models_by_ratio.items() if len(v) > 1}

    return {
        "config": {
            "attacks": ATTACKS, "difficulties": DIFFICULTIES,
            "min_signal_for_dprime": MIN_SIGNAL_FOR_DPRIME, "n_bootstrap": n_boot, "seed": seed,
            "metric_definitions": {
                "d_prime": "z(hit) - z(false_alarm); sensitivity",
                "criterion_c": "-0.5*(z(H)+z(FA)); >0 skeptical, <0 lenient",
                "auroc_margin": "rank AUROC of verdict logprob margin vs gold; threshold-free sensitivity",
                "false_positive_rate": "wrong-but-accepted / decided-wrong (lazy-accept rate)",
                "false_negative_rate": "correct-but-rejected / decided-correct",
                "correction": "Macmillan & Creelman log-linear (+0.5/+1) for extreme rates",
                "generator_accuracy": "count(gold True)/n_valid; verifier-independent",
                "system_accuracy_valid": "(tp+tn)/n_valid; abstentions penalized",
                "system_accuracy_decided": "(tp+tn)/n_decided; conditional on a verdict",
                "deception_success_rate": "lazy-accepted-wrong / all-wrong (abstain = NOT deceived); success+caught+abstain=1",
                "deception_success_rate_decided": "lazy-accepted-wrong / decided-wrong (== false_positive_rate)",
                "calibration.brier": "mean((p_correct - gold)^2)",
                "calibration.ece": "10-bin expected calibration error of p_correct vs gold",
                "calibration.confidence_gap": "mean p_correct on correct minus on incorrect generations",
            },
        },
        "meta": {
            "ratios": ratios,
            "n_total_valid": sum(per_ratio[f"ratio={r:.2f}"]["counts"]["n_valid"] for r in ratios),
            "models_seen": sorted(all_models),
            "models_by_ratio": {k: sorted(v) for k, v in models_by_ratio.items()},
            "model_mixing_within_ratio": mixing,
        },
        "correlations": {
            "pearson_gen_acc_vs_sys_acc": _pearson(gen_accs, sys_accs),
            "pearson_accept_rate_vs_sys_acc": _pearson(acc_rates, sys_accs),
            "pearson_ratio_vs_mismatch_fpr": _pearson(ratios, [mismatch_fpr[f"ratio={r:.2f}"] for r in ratios]),
        },
        "headline_findings": {
            "mismatch_fpr_by_ratio": mismatch_fpr,
            "mismatch_fpr_mean": _mean(mm_vals),
            "mismatch_fpr_range": [min(mm_vals), max(mm_vals)] if mm_vals else None,
            "contradiction_rate_by_ratio": {f"ratio={r:.2f}": v for r, v in zip(ratios, contra_vals)},
            "contradiction_rate_range": [min(contra_vals), max(contra_vals)] if contra_vals else None,
        },
        "flip_event_frequency": flip_events(by_ratio),
        "generator_token_usage": generator_token_usage(by_ratio, gen_by_id),
        "adequacy_warnings": adequacy_warnings,
        "per_ratio": per_ratio,
        "per_vbudget": per_vbudget,
    }


def print_summary(report: dict[str, Any]) -> None:
    print("=" * 104)
    print("  EXPERIMENT 2  -  VERIFIER BUDGET COLLAPSE  (unified metrics)")
    print(f"  ratios={report['meta']['ratios']}  N(valid)={report['meta']['n_total_valid']}  "
          f"models={report['meta']['models_seen']}")
    print("=" * 104)
    hdr = (f"{'ratio':>6} {'vmax':>6} {'n':>5} | {'gen':>6} {'sysV':>6} {'sysD':>6} {'EDR':>6} {'FPR':>6} "
           f"{'FNR':>6} {'DSR':>6} | {'d_prime':>8} {'crit_c':>8} {'AUROC':>6} | {'ECE':>6} {'contra':>7} {'mm_FPR':>7}")
    print(hdr)
    print("-" * len(hdr))
    f = lambda v, p=2, w=6: (f"{v:>{w}.{p}%}" if isinstance(v, (int, float)) else f"{'N/A':>{w}}")
    fn = lambda v, w=8: (f"{v:>{w}.3f}" if isinstance(v, (int, float)) else f"{'N/A':>{w}}")
    for r in report["meta"]["ratios"]:
        g = report["per_ratio"][f"ratio={r:.2f}"]
        co, acc, rt, dec, cal = (g["signal_detection"]["complete_only"], g["accuracy"],
                                 g["rates"], g["deception"], g["calibration"])
        mm = g["by_attack"].get("mismatch", {}).get("rates", {}).get("false_positive_rate")
        print(f"{r:>6.2f} {str(g['verifier_max_tokens_median'] or '-'):>6} {g['counts']['n_valid']:>5} | "
              f"{f(acc['generator_accuracy'])} {f(acc['system_accuracy_valid'])} {f(acc['system_accuracy_decided'])} "
              f"{f(rt['error_detection_rate'])} {f(rt['false_positive_rate'])} {f(rt['false_negative_rate'])} "
              f"{f(dec['deception_success_rate'])} | "
              f"{fn(co['d_prime'])} {fn(co['criterion_c'])} {fn(co['auroc_margin'], 6)} | "
              f"{fn(cal['ece'], 6)} {f(g['step_contradiction']['contradiction_rate'], 1, 7)} {f(mm, 1, 7)}")
    for w in report["adequacy_warnings"]:
        print(f"  ! {w}")
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2 (verifier budget collapse) analysis.")
    p.add_argument("--input-dir", default="data/experiments/verifier_budget_collapse/main")
    p.add_argument("--glob", default="*_vr*.jsonl")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    by_ratio = load_by_ratio(input_dir, args.glob)
    if not by_ratio:
        raise SystemExit(f"No result files found: {input_dir / args.glob}")
    gen_by_id = load_generator_rows(input_dir)

    report = analyze(by_ratio, gen_by_id, n_boot=args.n_bootstrap, seed=args.seed)
    fps, fns = collect_fp_fn(by_ratio)

    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment2_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_jsonl(fps, str(out_dir / "experiment2_false_positives.jsonl"))
    save_jsonl(fns, str(out_dir / "experiment2_false_negatives.jsonl"))

    print_summary(report)
    print(f"Wrote metrics : {out_dir / 'experiment2_metrics.json'}")
    print(f"Wrote FP      : {len(fps)} rows (wrong answers accepted)")
    print(f"Wrote FN      : {len(fns)} rows (correct answers rejected)")


if __name__ == "__main__":
    main()
