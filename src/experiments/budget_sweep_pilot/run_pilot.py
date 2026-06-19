from __future__ import annotations
import argparse
import asyncio
import json
import math
import re
import time
from pathlib import Path

from src.models.openai_runner import run_model_async
from src.parsers.verifier import parse_verifier
from src.prompts.verifier import build_verifier_prompt
from src.registry.configs import CONFIGS
from src.registry.datasets import DATASET_PATHS
from src.registry.evaluators import EVALUATOR_REGISTRY
from src.registry.parsers import GENERATION_PARSER_REGISTRY
from src.registry.prompts import GENERATOR_PROMPT_REGISTRY
from src.utils.io import load_jsonl, save_jsonl

_STEP_RE = re.compile(r"^\s*(\d+)\s*\.\s*(VALID|INVALID)\s*$", re.IGNORECASE)

def parse_step_judgments(output: str) -> dict:
    judgments = []
    for line in (output or "").splitlines():
        m = _STEP_RE.match(line)
        if m:
            judgments.append({"step_idx": int(m.group(1)) - 1, "judgment": m.group(2).upper()})
    if not judgments:
        return {"step_judgments": [], "total_steps": 0, "n_valid": 0, "n_invalid": 0, "first_invalid_idx": None}
    n_inv = sum(1 for j in judgments if j["judgment"] == "INVALID")
    first_inv = next((j["step_idx"] + 1 for j in judgments if j["judgment"] == "INVALID"), None)
    return {"step_judgments": judgments, "total_steps": len(judgments), "n_valid": len(judgments) - n_inv, "n_invalid": n_inv, "first_invalid_idx": first_inv}


def to_float(x):
    if x is None or isinstance(x, bool):
        return float(x) if isinstance(x, bool) else None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def p95(xs):
    return sorted(xs)[int(0.95 * (len(xs) - 1))] if xs else 0.0

def auroc(scores, labels):
    if len(scores) < 2 or len(set(labels)) < 2:
        return None

    data = list(zip(scores, labels))
    data.sort(key=lambda x: x[0], reverse=True)

    pos = [s for s, y in data if y == 1]
    neg = [s for s, y in data if y == 0]

    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None

    count = 0
    for p in pos:
        for n in neg:
            if p > n:
                count += 1
            elif p == n:
                count += 0.5

    return count / (n_pos * n_neg)

def save_json(data, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_existing(path):
    return {str(row["id"]): row for row in load_jsonl(path)}


def get_item_budget(difficulty: str, generator_ratio: float) -> dict:
    cfg = CONFIGS.get(difficulty, CONFIGS["medium"])
    total = cfg["total_max_tokens"]
    scale = cfg.get("generator_reasoning_budget_scale", 0.8)

    # Allocate strict proportions based on the sweep ratio
    gen = max(1, int(total * generator_ratio))
    ver = max(1, total - gen)

    return {
        "difficulty": difficulty,
        "total_max_tokens": total,
        "generator_ratio": float(generator_ratio),
        "generator_max_tokens": gen,
        "verifier_max_tokens": ver,
        "generator_reasoning_budget": int(gen * scale),
        "reasoning_budget_scale": scale,
    }


def _calibration_summary(probs, labels, n_bins=10):
    if not probs or len(probs) != len(labels):
        return {"n": 0, "brier": 0.0, "ece": 0.0}
    n = len(probs)
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n
    bins = [[] for _ in range(n_bins)]
    for i, p in enumerate(probs):
        bins[min(int(max(0.0, min(1.0, p)) * n_bins), n_bins - 1)].append(i)
    ece = 0.0
    for idxs in bins:
        if not idxs:
            continue
        avg_conf = mean([probs[i] for i in idxs])
        avg_acc = mean([float(labels[i]) for i in idxs])
        ece += (len(idxs) / n) * abs(avg_acc - avg_conf)
    return {"n": n, "brier": float(brier), "ece": float(ece)}


def _confidence_linkage(confs, labels):
    if not confs or not labels or len(confs) != len(labels):
        return {"gap": None}
    correct = [c for c, l in zip(confs, labels) if l == 1]
    incorrect = [c for c, l in zip(confs, labels) if l == 0]
    m_c = mean(correct) if correct else None
    m_i = mean(incorrect) if incorrect else None
    return {
        "mean_conf_correct": m_c,
        "mean_conf_incorrect": m_i,
        "gap": (m_c - m_i) if (m_c is not None and m_i is not None) else None,
    }

def _confidence_distribution(confs):
    if not confs:
        return {"mean": 0.0, "n": 0}
    n = len(confs)
    m = mean(confs)
    var = sum((p - m) ** 2 for p in confs) / n if n > 1 else 0.0
    thresholds = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    hist = {f"p>={t:.2f}": sum(1 for p in confs if p >= t) for t in thresholds}
    return {
        "mean": m,
        "variance": float(var),
        "std": float(var ** 0.5),
        "min": min(confs),
        "max": max(confs),
        "histogram": hist,
    }

def compute_confidence_and_calibration(rows):
    ver_pc, ver_yc, ver_pch, ver_yd, ver_cert = [], [], [], [], []
    gen_pp, gen_yc = [], []
    ver_avg_lp, ver_avg_ly = [], []

    for r in rows:
        actual = r.get("actual_correctness")
        y = 1 if actual is True else 0 if actual is False else None

        pc = to_float(r.get("verifier_verdict_p_correct"))
        if pc is not None and y is not None:
            ver_pc.append(pc); ver_yc.append(y)

        cert = to_float(r.get("verifier_verdict_certainty"))
        if cert is not None:
            ver_cert.append(cert)

        pch = to_float(r.get("verifier_verdict_p_chosen"))
        dc = r.get("verifier_correct")
        if pch is not None and dc in (True, False):
            ver_pch.append(pch); ver_yd.append(1 if dc else 0)

        glp = to_float(r.get("generator_completion_avg_logprob"))
        if glp is not None and y is not None:
            gen_yc.append(y); gen_pp.append(math.exp(glp))

        valp = to_float(r.get("verifier_completion_avg_logprob"))
        if valp is not None and y is not None:
            ver_avg_ly.append(y); ver_avg_lp.append(valp)

    ver_conf = {
        "mean_p_correct": mean(ver_pc),
        "mean_p_chosen": mean(ver_pch),
        "mean_certainty": mean(ver_cert),
    }
    gen_conf = {"mean_token_prob_proxy": mean(gen_pp)}
    ver_avg_conf = {"mean_avg_logprob": mean(ver_avg_lp)}

    ver_linkage = _confidence_linkage(ver_pc, ver_yc) if ver_pc else {"gap": None}
    gen_linkage = _confidence_linkage(gen_pp, gen_yc) if gen_pp else {"gap": None}
    ver_avg_linkage = _confidence_linkage(ver_avg_lp, ver_avg_ly) if ver_avg_lp else {"gap": None}

    ver_auroc_val = auroc(ver_pc, ver_yc) if ver_pc else None
    gen_auroc_val = auroc(gen_pp, gen_yc) if gen_pp else None
    ver_avg_auroc_val = auroc(ver_avg_lp, ver_avg_ly) if ver_avg_lp else None

    ver_dist = _confidence_distribution(ver_pc)

    cal_by_split = {}
    if ver_pc:
        corr = [p for p, y in zip(ver_pc, ver_yc) if y == 1]
        incorr = [p for p, y in zip(ver_pc, ver_yc) if y == 0]
        cal_by_split["verifier_p_correct"] = {
            "correct": {"mean": mean(corr), "n": len(corr)},
            "incorrect": {"mean": mean(incorr), "n": len(incorr)},
            "gap": mean(corr) - mean(incorr) if corr and incorr else None,
        }
    if gen_pp:
        gcorr = [p for p, y in zip(gen_pp, gen_yc) if y == 1]
        gincorr = [p for p, y in zip(gen_pp, gen_yc) if y == 0]
        cal_by_split["generator"] = {
            "correct": {"mean": mean(gcorr), "n": len(gcorr)},
            "incorrect": {"mean": mean(gincorr), "n": len(gincorr)},
            "gap": mean(gcorr) - mean(gincorr) if gcorr and gincorr else None,
        }

    return {
        "confidence": {
            "verifier": ver_conf,
            "generator": gen_conf,
            "verifier_avg_logprob": ver_avg_conf,
        },
        "calibration": {
            "verifier_p_correct": _calibration_summary(ver_pc, ver_yc),
            "verifier_p_chosen": _calibration_summary(ver_pch, ver_yd),
            "generator_token_prob_proxy": _calibration_summary(gen_pp, gen_yc),
        },
        "confidence_accuracy_linkage": {
            "verifier_p_correct": ver_linkage,
            "generator_token_prob_proxy": gen_linkage,
            "verifier_avg_logprob": ver_avg_linkage,
        },
        "discrimination": {
            "verifier_p_correct_auroc": ver_auroc_val,
            "generator_token_prob_proxy_auroc": gen_auroc_val,
            "verifier_avg_logprob_auroc": ver_avg_auroc_val,
        },
        "calibration_by_split": cal_by_split,
        "verifier_confidence_distribution": ver_dist,
    }


def compute_verdict_probability_metrics(token_logprobs, chosen_verdict=None):
    if not isinstance(token_logprobs, list) or not token_logprobs:
        return {}

    def norm(t):
        return str(t).strip().upper() if t is not None else ""

    verdict_entry = next(
        (e for e in reversed(token_logprobs)
         if isinstance(e, dict) and norm(e.get("token")) in {"CORRECT", "INCORRECT"}),
        None,
    )
    if verdict_entry is None:
        return {}

    actual_lp = to_float(verdict_entry.get("logprob"))
    actual_token = norm(verdict_entry.get("token"))

    lp = {}
    for alt in verdict_entry.get("top_logprobs") or []:
        tok = norm(alt.get("token"))
        try:
            lp[tok] = float(alt["logprob"])
        except (KeyError, TypeError, ValueError):
            pass

    if "CORRECT" in lp and "INCORRECT" in lp:
        p_c, p_i = math.exp(lp["CORRECT"]), math.exp(lp["INCORRECT"])
        total = p_c + p_i
        if total > 0:
            p_c, p_i = p_c / total, p_i / total

            def entropy_bits(p):
                return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

            entropy = entropy_bits(p_c)
            chosen = norm(chosen_verdict)
            p_chosen = p_other = margin_lp = margin_p = None
            if chosen in {"CORRECT", "INCORRECT"}:
                other = "INCORRECT" if chosen == "CORRECT" else "CORRECT"
                p_chosen = p_c if chosen == "CORRECT" else p_i
                p_other  = p_i if chosen == "CORRECT" else p_c
                margin_lp = lp[chosen] - lp[other]
                margin_p  = p_chosen - p_other

            return {
                "verifier_verdict_lp_correct": lp["CORRECT"],
                "verifier_verdict_lp_incorrect": lp["INCORRECT"],
                "verifier_verdict_p_correct": p_c,
                "verifier_verdict_p_incorrect": p_i,
                "verifier_verdict_p_chosen": p_chosen,
                "verifier_verdict_p_other": p_other,
                "verifier_verdict_entropy_bits": entropy,
                "verifier_verdict_certainty": 1.0 - entropy,
                "verifier_verdict_margin_logprob": lp["CORRECT"] - lp["INCORRECT"],
                "verifier_verdict_margin_prob": p_c - p_i,
                "verifier_verdict_margin_logprob_chosen_minus_other": margin_lp,
                "verifier_verdict_margin_prob_chosen_minus_other": margin_p,
            }

    if actual_lp is not None:
        p_correct = math.exp(actual_lp)
        certainty = p_correct
        if actual_token == "CORRECT":
            p_correct = min(p_correct, 1.0)
        elif actual_token == "INCORRECT":
            p_correct = max(1.0 - p_correct, 0.0)
        else:
            return {}
        return {
            "verifier_verdict_p_correct": float(p_correct),
            "verifier_verdict_certainty": float(certainty),
        }

    return {}


def build_result_row(
    item, gen_resp, ver_resp, gen_reasoning, predicted,
    actual_correctness, ver_decision, dataset, model, budget,
):
    gen_finish = gen_resp.get("finish_reason")
    ver_finish = ver_resp.get("finish_reason")
    chosen_verdict = "CORRECT" if ver_decision is True else "INCORRECT" if ver_decision is False else None
    verdict_probs = compute_verdict_probability_metrics(
        ver_resp.get("token_logprobs"), chosen_verdict=chosen_verdict
    )
    verifier_correct = ver_decision is not None and ver_decision == actual_correctness
    step_info = parse_step_judgments(ver_resp.get("text", ""))
    return {
        "id": item.get("id"), "dataset": dataset, "model": model,
        "gold": item.get("answer", {}).get("ideal"),
        "predicted_answer": predicted,
        "actual_correctness": actual_correctness, "generator_correct": actual_correctness,
        "truncated": gen_finish == "length",
        "generator_raw_output": gen_resp.get("text"), "generator_finish_reason": gen_finish,
        "generator_max_tokens": budget["generator_max_tokens"],
        "generator_reasoning_budget": budget["generator_reasoning_budget"],
        "generator_reasoning": gen_reasoning,
        "prompt_tokens": gen_resp.get("prompt_tokens"),
        "completion_tokens": gen_resp.get("completion_tokens"),
        "total_tokens_used": gen_resp.get("total_tokens"),
        "generator_completion_avg_logprob": gen_resp.get("completion_avg_logprob"),
        "verifier_completion_avg_logprob": ver_resp.get("completion_avg_logprob"),
        "verifier_raw_output": ver_resp.get("text"), "verifier_finish_reason": ver_finish,
        "verifier_truncated": ver_finish == "length",
        "verifier_max_tokens": budget["verifier_max_tokens"],
        "verifier_prompt_tokens": ver_resp.get("prompt_tokens"),
        "verifier_completion_tokens": ver_resp.get("completion_tokens"),
        "verifier_total_tokens_used": ver_resp.get("total_tokens"),
        "verifier_decision": ver_decision, "verifier_correct": verifier_correct,
        "verifier_lazy_accept": ver_decision is True and not actual_correctness,
        "verifier_false_reject": ver_decision is False and actual_correctness,
        "verifier_abstained": ver_decision is None,
        **verdict_probs,
        "step_judgments": step_info["step_judgments"],
        "total_steps": step_info["total_steps"],
        "n_valid_steps": step_info["n_valid"],
        "n_invalid_steps": step_info["n_invalid"],
        "first_invalid_step": step_info["first_invalid_idx"],
        "budget": budget,
        "usage": {
            "generator": {
                "prompt_tokens": gen_resp.get("prompt_tokens"),
                "completion_tokens": gen_resp.get("completion_tokens"),
                "total_tokens": gen_resp.get("total_tokens"),
                "completion_avg_logprob": gen_resp.get("completion_avg_logprob"),
                "completion_logprob_sum": gen_resp.get("completion_logprob_sum"),
                "completion_logprob_count": gen_resp.get("completion_logprob_count"),
                "model": gen_resp.get("model"),
                "finish_reason": gen_finish,
                "response_id": gen_resp.get("response_id"),
            },
            "verifier": {
                "prompt_tokens": ver_resp.get("prompt_tokens"),
                "completion_tokens": ver_resp.get("completion_tokens"),
                "total_tokens": ver_resp.get("total_tokens"),
                "completion_avg_logprob": ver_resp.get("completion_avg_logprob"),
                "completion_logprob_sum": ver_resp.get("completion_logprob_sum"),
                "completion_logprob_count": ver_resp.get("completion_logprob_count"),
                "model": ver_resp.get("model"),
                "finish_reason": ver_finish,
                "response_id": ver_resp.get("response_id"),
            },
        },
    }


def _per_difficulty_stats(valid: list[dict]) -> dict:
    groups: dict[str, list] = {}
    for r in valid:
        d = r.get("budget", {}).get("difficulty", "unknown")
        groups.setdefault(d, []).append(r)

    out = {}
    for diff, rows in groups.items():
        n = len(rows)
        gen_acc = sum(1 for r in rows if r.get("actual_correctness")) / n if n else 0.0
        
        correct_system = 0
        for r in rows:
            v_dec = r.get("verifier_decision")
            act_corr = r.get("actual_correctness")
            if v_dec is True and act_corr is True:
                correct_system += 1
            elif v_dec is False and act_corr is False:
                correct_system += 1
            elif v_dec is None:
                if act_corr is True:
                    correct_system += 1

        sys_acc = correct_system / n if n else 0.0
        gen_tokens = [r.get("completion_tokens") or 0 for r in rows]
        step_rows = [r for r in rows if r.get("total_steps", 0) > 0]
        tp = sum(r["n_invalid_steps"] for r in step_rows if not r.get("actual_correctness"))
        fn = sum(r["n_valid_steps"] for r in step_rows if not r.get("actual_correctness"))
        fp = sum(r["n_invalid_steps"] for r in step_rows if r.get("actual_correctness"))
        tn = sum(r["n_valid_steps"] for r in step_rows if r.get("actual_correctness"))
        total_steps = tp + fn + fp + tn
        step_acc = (tp + tn) / total_steps if total_steps else None
        step_recall = tp / (tp + fn) if (tp + fn) else None
        step_fp_rate = fp / (fp + tn) if (fp + tn) else None
        consistency = sum(
            1 for r in step_rows
            if (r.get("verifier_decision") is True and r["n_invalid_steps"] == 0)
            or (r.get("verifier_decision") is False and r["n_invalid_steps"] > 0)
        )
        first_errors = [r["first_invalid_step"] for r in step_rows if r["first_invalid_step"] is not None]
        out[diff] = {
            "n": n,
            "generator_accuracy": gen_acc,
            "system_accuracy": sys_acc,
            "accuracy_delta": sys_acc - gen_acc,
            "generator_truncation_rate": sum(1 for r in rows if r.get("truncated")) / n if n else 0.0,
            "avg_generator_tokens_used": mean(gen_tokens),
            "step_accuracy": step_acc,
            "step_recall": step_recall,
            "step_fp_rate": step_fp_rate,
            "consistency_rate": consistency / len(step_rows) if step_rows else None,
            "avg_invalid_steps": mean([r["n_invalid_steps"] for r in step_rows]) if step_rows else None,
            "error_position": mean(first_errors) if first_errors else None,
        }
    return out


def _compute_decision_simulation(valid):
    decided = [r for r in valid if r.get("verifier_decision") is not None]
    if not decided:
        return {"accept_top_k_percent": {}}
    scored = []
    for r in decided:
        alp = to_float(r.get("verifier_completion_avg_logprob"))
        if alp is not None:
            correct = r.get("actual_correctness") is True and r.get("verifier_decision") is True
            scored.append((alp, correct))
    if not scored:
        return {"accept_top_k_percent": {}}
    scored.sort(key=lambda x: -x[0])
    total = len(scored)
    n_valid = len(valid)
    result = {}
    for k in [10, 25, 50, 75, 90]:
        n_accept = max(1, int(total * k / 100))
        top = scored[:n_accept]
        n_correct = sum(1 for _, ok in top if ok)
        result[f"k={k}"] = {
            "system_accuracy": n_correct / n_accept,
            "accept_rate": n_accept / n_valid if n_valid else 0.0,
        }
    return {"accept_top_k_percent": result}


def fnr_by_step_count(valid):
    decided = [r for r in valid if r.get("verifier_decision") is not None and r.get("total_steps", 0) > 0]
    if not decided:
        return {}
    bins = {"1-3": [], "4-6": [], "7+": []}
    for r in decided:
        s = r["total_steps"]
        key = "7+" if s >= 7 else "4-6" if s >= 4 else "1-3"
        bins[key].append(r)
    result = {}
    for label, group in bins.items():
        if not group:
            result[label] = {"n": 0, "fnr": None}
            continue
        fr = sum(1 for r in group if r.get("verifier_decision") is False and r.get("actual_correctness"))
        ca = sum(1 for r in group if r.get("verifier_decision") is True and r.get("actual_correctness"))
        denom = fr + ca
        result[label] = {"n": len(group), "fnr": fr / denom if denom else None}
    return result


def compute_summary(results, dataset, model, temperature, generator_ratio):
    valid = [r for r in results if "error" not in r]
    n = len(valid)

    decided = [r for r in valid if r.get("verifier_decision") is not None]
    nd = len(decided)
    correct_accepts = sum(1 for r in decided if r.get("actual_correctness") is True  and r.get("verifier_decision") is True)
    false_rejects   = sum(1 for r in decided if r.get("actual_correctness") is True  and r.get("verifier_decision") is False)
    correct_rejects = sum(1 for r in decided if r.get("actual_correctness") is False and r.get("verifier_decision") is False)
    lazy_accepts    = sum(1 for r in decided if r.get("actual_correctness") is False and r.get("verifier_decision") is True)
    fpr_denom = lazy_accepts + correct_rejects
    fnr_denom = false_rejects + correct_accepts 

    gen_accuracy = sum(1 for r in valid if r.get("actual_correctness")) / n if n else 0.0
    
    correct_system_total = 0
    for r in valid:
        v_dec = r.get("verifier_decision")
        act_corr = r.get("actual_correctness")
        if v_dec is True and act_corr is True:
            correct_system_total += 1
        elif v_dec is False and act_corr is False:
            correct_system_total += 1
        elif v_dec is None and act_corr is True:
            correct_system_total += 1
            
    sys_accuracy = correct_system_total / n if n else 0.0

    difficulty = _per_difficulty_stats(valid)

    gen_tokens_used = [r.get("completion_tokens") or 0 for r in valid]
    ver_tokens_used = [r.get("verifier_completion_tokens") or 0 for r in valid]
    gen_utils = [
        r.get("completion_tokens", 0) / r["budget"]["generator_max_tokens"]
        for r in valid if r.get("budget", {}).get("generator_max_tokens")
    ]
    ver_utils = [
        (r.get("verifier_completion_tokens") or 0) / r["budget"]["verifier_max_tokens"]
        for r in valid if r.get("budget", {}).get("verifier_max_tokens")
    ]

    non_trunc = [r for r in valid if not r.get("truncated")]

    per_diff_budget = {
        diff: {k: v for k, v in get_item_budget(diff, generator_ratio).items() if k != "difficulty"}
        for diff in difficulty
        if diff != "unknown"
    }

    conf = compute_confidence_and_calibration(valid)

    step_rows = [r for r in valid if r.get("total_steps", 0) > 0]
    tp = sum(r["n_invalid_steps"] for r in step_rows if not r.get("actual_correctness"))
    fn = sum(r["n_valid_steps"] for r in step_rows if not r.get("actual_correctness"))
    fp = sum(r["n_invalid_steps"] for r in step_rows if r.get("actual_correctness"))
    tn = sum(r["n_valid_steps"] for r in step_rows if r.get("actual_correctness"))
    total_judged = tp + fn + fp + tn
    step_acc = (tp + tn) / total_judged if total_judged else None
    step_recall = tp / (tp + fn) if (tp + fn) else None
    step_fp_rate = fp / (fp + tn) if (fp + tn) else None
    consistency = sum(
        1 for r in step_rows
        if (r.get("verifier_decision") is True and r["n_invalid_steps"] == 0)
        or (r.get("verifier_decision") is False and r["n_invalid_steps"] > 0)
    )
    first_errors = [r["first_invalid_step"] for r in step_rows if r["first_invalid_step"] is not None]
    return {
        "meta": {
            "dataset": dataset,
            "model": model,
            "n_valid": n,
        },
        "budget": {
            "generator_ratio": float(generator_ratio),
            "per_difficulty": per_diff_budget,
        },
        "performance": {
            "overall": {
                "generator_accuracy": gen_accuracy,
                "system_accuracy": sys_accuracy,
                "accuracy_delta": sys_accuracy - gen_accuracy,
            },
            "by_difficulty": difficulty,
        },
        "verification": {
            "summary": {
                "verifier_accuracy": sum(1 for r in valid if r.get("verifier_correct") is True) / n if n else 0.0,
                "accept_rate": (correct_accepts + lazy_accepts) / nd if nd else 0.0,
                "reject_rate": (correct_rejects + false_rejects) / nd if nd else 0.0,
                "error_detection_rate": correct_rejects / fpr_denom if fpr_denom else 0.0,
                "false_positive_rate": lazy_accepts / fpr_denom if fpr_denom else 0.0,
                "false_negative_rate": false_rejects / fnr_denom if fnr_denom else 0.0,
            },
            "step_level": {
                "step_accuracy": step_acc,
                "step_recall": step_recall,
                "step_fp_rate": step_fp_rate,
                "consistency_rate": consistency / len(step_rows) if step_rows else None,
                "avg_invalid_steps": mean([r["n_invalid_steps"] for r in step_rows]) if step_rows else None,
                "error_position": mean(first_errors) if first_errors else None,
            },
            "fnr_by_step_count": fnr_by_step_count(valid),
        },
        "token_usage": {
            "generator": {
                "mean_used": mean(gen_tokens_used),
                "utilization": mean(gen_utils),
            },
            "verifier": {
                "mean_used": mean(ver_tokens_used),
                "utilization": mean(ver_utils),
            },
        },
        "truncation": {
            "generator_truncation_rate": sum(1 for r in valid if r.get("truncated")) / n if n else 0.0,
            "verifier_truncation_rate": sum(1 for r in valid if r.get("verifier_truncated")) / n if n else 0.0,
            "generator_accuracy_nontruncated": sum(1 for r in non_trunc if r.get("actual_correctness")) / len(non_trunc) if non_trunc else 0.0,
        },
        "confidence": {
            "generator": {
                "mean_token_prob_proxy": conf["confidence"]["generator"]["mean_token_prob_proxy"],
            },
            "verifier": {
                "logprob_proxy": {
                    "mean": conf["confidence"]["verifier_avg_logprob"]["mean_avg_logprob"],
                },
            },
        },
        "confidence_linkage": {
            "generator": {
                "confidence_correct_gap": conf["confidence_accuracy_linkage"]["generator_token_prob_proxy"]["gap"],
                "mean_conf_correct": conf["confidence_accuracy_linkage"]["generator_token_prob_proxy"]["mean_conf_correct"],
                "mean_conf_incorrect": conf["confidence_accuracy_linkage"]["generator_token_prob_proxy"]["mean_conf_incorrect"],
            },
            "verifier": {
                "confidence_correct_gap": conf["confidence_accuracy_linkage"]["verifier_avg_logprob"]["gap"],
                "mean_conf_correct": conf["confidence_accuracy_linkage"]["verifier_avg_logprob"]["mean_conf_correct"],
                "mean_conf_incorrect": conf["confidence_accuracy_linkage"]["verifier_avg_logprob"]["mean_conf_incorrect"],
            },
        },
        "discrimination": {
            "generator": {
                "auroc": conf["discrimination"]["generator_token_prob_proxy_auroc"],
            },
            "verifier": {
                "auroc": conf["discrimination"]["verifier_avg_logprob_auroc"],
            },
        },
        "calibration": {
            "generator": {
                "brier": conf["calibration"]["generator_token_prob_proxy"]["brier"],
                "ece": conf["calibration"]["generator_token_prob_proxy"]["ece"],
            },
        },
        "decision_policy_simulation": _compute_decision_simulation(valid),
    }


async def run_dataset_ratio_async(
    input_path, dataset, generator_ratio,
    model, temperature, limit, out_path, save_every,
    concurrency, batch_size, retry_errors_only,
    prompt_builder, parse_generation, evaluator,
    *, verifier_provider="openai", verifier_model=None,
):
    sem = asyncio.Semaphore(concurrency)
    existing = load_existing(out_path) if retry_errors_only else {}

    n_total, to_run, cached = 0, [], {}
    for idx, item in enumerate(load_jsonl(input_path)):
        if limit is not None and idx >= limit:
            break
        n_total += 1
        key = str(item.get("id", f"__idx__{idx}"))
        if retry_errors_only and existing.get(key) and not existing[key].get("error"):
            cached[idx] = existing[key]
        else:
            to_run.append((idx, item))

    async def process_one(item):
        budget = get_item_budget(item.get("difficulty", "medium"), generator_ratio)
        gen_prompt = prompt_builder(item, reasoning_budget=budget["generator_reasoning_budget"])
        async with sem:
            gen_resp = await run_model_async(
                prompt=gen_prompt, model=model, temperature=temperature,
                max_tokens=budget["generator_max_tokens"], logprobs=True,
                provider="openai",
            )
        gen_reasoning, predicted = parse_generation(gen_resp["text"])
        actual_correctness = evaluator(predicted, item)
        ver_prompt = build_verifier_prompt(
            question=item.get("question", ""), reasoning=gen_reasoning or "", answer=predicted
        )
        async with sem:
            ver_resp = await run_model_async(
                prompt=ver_prompt, model=verifier_model or model, temperature=temperature,
                max_tokens=budget["verifier_max_tokens"], logprobs=True,
                top_logprobs=50, return_token_logprobs=True, token_logprobs_last_n=12,
                provider=verifier_provider,
            )
        ver_decision = parse_verifier(ver_resp["text"])
        return build_result_row(
            item, gen_resp, ver_resp, gen_reasoning, predicted,
            actual_correctness, ver_decision, dataset, model, budget,
        )

    out_by_idx = dict(cached)
    for start in range(0, len(to_run), batch_size):
        batch = to_run[start : start + batch_size]
        rows = await asyncio.gather(*[asyncio.create_task(process_one(item)) for _, item in batch])
        for (idx, _), row in zip(batch, rows):
            out_by_idx[idx] = row
        if save_every and len(out_by_idx) % save_every == 0:
            save_jsonl([out_by_idx[i] for i in range(n_total) if i in out_by_idx], out_path)

    return [
        out_by_idx.get(i, {
            "id": None, "dataset": dataset,
            "model": model, "error": "Missing result row",
        })
        for i in range(n_total)
    ]


def run_sweep(
    *, datasets, model, temperature, limit,
    output_dir, save_every, generator_ratios,
    concurrency=1, batch_size=None, retry_errors_only=False, input_override=None,
    verifier_provider="openai", verifier_model=None,
):
    concurrency = max(1, concurrency)
    batch_size = batch_size or max(10, concurrency * 5)
    
    if generator_ratios is None:
        generator_ratios = CONFIGS["medium"]["generator_ratios"] # same for all difficulty levels 

    for dataset in datasets:
        ds_paths = DATASET_PATHS.get(dataset) or {}
        input_path = input_override or next(
            (ds_paths[k] for k in ("sample", "raw", "pilot") if ds_paths.get(k)),
            None,
        )
        if not input_path:
            print(f"ERROR: no input path for dataset {dataset!r}")
            continue
        prompt_builder = GENERATOR_PROMPT_REGISTRY[dataset]
        parse_generation = GENERATION_PARSER_REGISTRY[dataset]
        evaluator = EVALUATOR_REGISTRY[dataset]

        for ratio in generator_ratios:
            out_path = f"{output_dir}/{dataset}_r{int(ratio * 100)}.jsonl"

            results = asyncio.run(run_dataset_ratio_async(
                input_path, dataset, ratio,
                model, temperature, limit, out_path, save_every,
                concurrency, batch_size, retry_errors_only,
                prompt_builder, parse_generation, evaluator,
                verifier_provider=verifier_provider, verifier_model=verifier_model,
            ))
            save_jsonl(results, out_path)

            summary = compute_summary(results, dataset, model, temperature, ratio)
            save_json(summary, out_path.replace(".jsonl", "_summary.json"))

            ov = summary["performance"]["overall"]
            print(
                f"[{dataset}] ratio={ratio:.2f} | "
                f"gen={ov['generator_accuracy']:.2%} "
                f"sys={ov['system_accuracy']:.2%} "
                f"delta={ov['accuracy_delta']:+.2%}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sweep over Generator → Verifier configurations.")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "strategyqa", "truthfulqa"])
    parser.add_argument("--generator-ratios", type=float, nargs="+", default=None)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="data/experiments/sweep")
    parser.add_argument("--input", dest="input_override", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--retry-errors-only", action="store_true")
    parser.add_argument("--verifier-provider", default="openai", choices=["openai", "qwen", "llama"],
                        help="API provider for the verifier model (default: openai)")
    parser.add_argument("--verifier-model", default=None,
                        help="Verifier model (auto-set to Qwen/Qwen2.5-7B-Instruct-Turbo when --verifier-provider qwen)")
    return parser.parse_args()


if __name__ == "__main__":
    t0 = time.perf_counter()
    args = parse_args()
    if args.verifier_provider == "qwen" and args.verifier_model is None:
        args.verifier_model = "Qwen/Qwen2.5-7B-Instruct-Turbo"
    run_sweep(
        datasets=args.datasets,
        model=args.model,
        temperature=args.temperature,
        limit=args.limit,
        output_dir=args.output_dir,
        save_every=args.save_every,
        generator_ratios=args.generator_ratios,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        retry_errors_only=args.retry_errors_only,
        input_override=args.input_override,
        verifier_provider=args.verifier_provider,
        verifier_model=args.verifier_model,
    )
    
    elapsed = time.perf_counter() - t0
    hh, rem = divmod(int(elapsed), 3600)
    mm, ss = divmod(rem, 60)
    print(f"Total runtime: {hh:02d}:{mm:02d}:{ss:02d} ({elapsed:.2f}s)")
