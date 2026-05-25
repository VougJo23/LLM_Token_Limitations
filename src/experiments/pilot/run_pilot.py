from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from src.models.openai_runner import run_model, run_model_async
from src.parsers.verifier import parse_verifier
from src.prompts.verifier import build_verifier_prompt
from src.registry.configs import CONFIGS, PILOT_CONFIGS
from src.registry.datasets import DATASET_PATHS
from src.registry.evaluators import EVALUATOR_REGISTRY
from src.registry.parsers import GENERATION_PARSER_REGISTRY
from src.registry.prompts import GENERATOR_PROMPT_REGISTRY
from src.utils.io import load_jsonl, save_jsonl


def _compute_verdict_probability_metrics(
    token_logprobs: Any,
    *,
    chosen_verdict: str | None = None,
) -> dict[str, Any]:
    if not isinstance(token_logprobs, list) or not token_logprobs:
        return {}

    def _norm_token(t: Any) -> str:
        return ("" if t is None else str(t)).strip().upper()

    verdict_entry: dict[str, Any] | None = None
    for entry in reversed(token_logprobs):
        if not isinstance(entry, dict):
            continue
        tok = _norm_token(entry.get("token"))
        if tok in {"CORRECT", "INCORRECT"}:
            verdict_entry = entry
            break

    if not verdict_entry:
        return {}

    top = verdict_entry.get("top_logprobs")
    if not isinstance(top, list) or not top:
        return {}

    lp_correct: float | None = None
    lp_incorrect: float | None = None
    for alt in top:
        if not isinstance(alt, dict):
            continue
        alt_tok = _norm_token(alt.get("token"))
        alt_lp = alt.get("logprob")
        try:
            alt_lp_f = float(alt_lp) if alt_lp is not None else None
        except Exception:
            alt_lp_f = None

        if alt_tok == "CORRECT":
            lp_correct = alt_lp_f
        elif alt_tok == "INCORRECT":
            lp_incorrect = alt_lp_f

    if lp_correct is None or lp_incorrect is None:
        return {}

    p_correct = math.exp(lp_correct)
    p_incorrect = math.exp(lp_incorrect)
    denom = p_correct + p_incorrect
    if denom <= 0:
        return {}

    pc = p_correct / denom
    pi = p_incorrect / denom

    def _entropy_bits(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -(p * math.log(p, 2) + (1.0 - p) * math.log(1.0 - p, 2))

    entropy_bits = _entropy_bits(pc)
    certainty = 1.0 - entropy_bits

    chosen = ("" if chosen_verdict is None else str(chosen_verdict)).strip().upper()
    p_chosen: float | None = None
    p_other: float | None = None
    margin_lp_chosen_minus_other: float | None = None
    margin_p_chosen_minus_other: float | None = None
    if chosen in {"CORRECT", "INCORRECT"}:
        if chosen == "CORRECT":
            p_chosen, p_other = pc, pi
            margin_lp_chosen_minus_other = lp_correct - lp_incorrect
            margin_p_chosen_minus_other = pc - pi
        else:
            p_chosen, p_other = pi, pc
            margin_lp_chosen_minus_other = lp_incorrect - lp_correct
            margin_p_chosen_minus_other = pi - pc

    return {
        "verifier_verdict_lp_correct": lp_correct,
        "verifier_verdict_lp_incorrect": lp_incorrect,
        "verifier_verdict_p_correct": pc,
        "verifier_verdict_p_incorrect": pi,
        "verifier_verdict_p_chosen": p_chosen,
        "verifier_verdict_p_other": p_other,
        "verifier_verdict_entropy_bits": entropy_bits,
        "verifier_verdict_certainty": certainty,
        "verifier_verdict_margin_logprob": lp_correct - lp_incorrect,
        "verifier_verdict_margin_prob": pc - pi,
        "verifier_verdict_margin_logprob_chosen_minus_other": margin_lp_chosen_minus_other,
        "verifier_verdict_margin_prob_chosen_minus_other": margin_p_chosen_minus_other,
    }


def _allocate_budgets(
    total_max_tokens: int,
    generator_ratio: float,
    verifier_min_tokens: int = 1,
) -> tuple[int, int]:
    if total_max_tokens < 2:
        return 1, 1

    verifier_min_tokens = max(1, int(verifier_min_tokens))

    gen = max(1, int(total_max_tokens * generator_ratio))
    ver = max(1, total_max_tokens - gen)

    if ver < verifier_min_tokens:
        ver = verifier_min_tokens
        gen = total_max_tokens - ver

    if gen < 1:
        gen = 1
        ver = max(1, total_max_tokens - gen)
    return gen, ver


def _save_json(data: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pilot(
    *,
    datasets: list[str],
    pilot_config_name: str,
    gen_config_name: str,
    model: str,
    temperature: float,
    limit: int | None,
    output_dir: str,
    save_every: int,
    total_max_tokens: int | None = None,
    generator_reasoning_budget: int | None = None,
    generator_ratios: list[float] | None = None,
    write_manifest: bool = False,
    concurrency: int = 1,
    batch_size: int | None = None,
) -> None:
    pilot_cfg = PILOT_CONFIGS[pilot_config_name]
    gen_cfg = CONFIGS[gen_config_name]

    total_max_tokens = int(total_max_tokens) if total_max_tokens is not None else int(pilot_cfg["total_max_tokens"])
    generator_ratios = list(generator_ratios) if generator_ratios is not None else list(pilot_cfg["generator_ratios"])
    verifier_min_tokens = int(pilot_cfg.get("verifier_min_tokens", 1))
    reasoning_budget_scale = float(pilot_cfg.get("generator_reasoning_budget_scale", 0.8))

    target_reasoning_budget = (
        int(generator_reasoning_budget)
        if generator_reasoning_budget is not None
        else int(gen_cfg["reasoning_budget"])
    )

    run_started_at = time.time()
    created_outputs: list[dict[str, str]] = []

    concurrency = max(1, int(concurrency))
    batch_size = int(batch_size) if batch_size is not None else max(10, concurrency * 5)

    def _iter_batches(items: list[tuple[int, dict[str, Any]]]):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    for dataset in datasets:
        input_path = DATASET_PATHS[dataset]["pilot"]
        prompt_builder = GENERATOR_PROMPT_REGISTRY[dataset]
        parse_generation = GENERATION_PARSER_REGISTRY[dataset]
        evaluator = EVALUATOR_REGISTRY[dataset]

        for ratio in generator_ratios:
            generator_max_tokens, verifier_max_tokens = _allocate_budgets(
                total_max_tokens=total_max_tokens,
                generator_ratio=float(ratio),
                verifier_min_tokens=verifier_min_tokens,
            )

            # Prompt-level reasoning budget (soft). 
            # Actual generator max tokens is a hard cap that may be hit before this if the model outputs a lot of reasoning.
            generator_reasoning_budget = min(
                int(target_reasoning_budget),
                max(1, int(generator_max_tokens * reasoning_budget_scale)),
            )

            results: list[dict[str, Any]] = []

            out_path = f"{output_dir}/{dataset}_{pilot_config_name}_{gen_config_name}_r{int(float(ratio) * 100)}.jsonl"

            if concurrency <= 1:
                for idx, item in enumerate(load_jsonl(input_path)):
                    if limit is not None and idx >= limit:
                        break

                    try:
                        gen_prompt = prompt_builder(item, reasoning_budget=generator_reasoning_budget)
                        gen_resp = run_model(
                            prompt=gen_prompt,
                            model=model,
                            temperature=temperature,
                            max_tokens=generator_max_tokens,
                            logprobs=True,
                        )

                        gen_text = gen_resp["text"]
                        gen_finish_reason = gen_resp.get("finish_reason")
                        gen_truncated = gen_finish_reason == "length"
                        gen_reasoning, predicted = parse_generation(gen_text)

                        actual_correctness = evaluator(predicted, item)

                        ver_prompt = build_verifier_prompt(
                            question=item.get("question", ""),
                            reasoning=gen_reasoning or "",
                            answer=predicted,
                        )

                        ver_resp = run_model(
                            prompt=ver_prompt,
                            model=model,
                            temperature=temperature,
                            max_tokens=verifier_max_tokens,
                            logprobs=True,
                            top_logprobs=50,
                            return_token_logprobs=True,
                            token_logprobs_last_n=12,
                        )

                        ver_text = ver_resp["text"]
                        ver_finish_reason = ver_resp.get("finish_reason")
                        ver_truncated = ver_finish_reason == "length"
                        ver_decision = parse_verifier(ver_text)

                        chosen_verdict = "CORRECT" if ver_decision is True else "INCORRECT" if ver_decision is False else None
                        verdict_prob_metrics = _compute_verdict_probability_metrics(
                            ver_resp.get("token_logprobs"),
                            chosen_verdict=chosen_verdict,
                        )

                        verifier_correct = ver_decision is not None and ver_decision == actual_correctness

                        gold = item.get("answer", {}).get("ideal")

                        result = {
                            "id": item.get("id"),
                            "dataset": dataset,
                            "pilot_config": pilot_config_name,
                            "config": gen_config_name,
                            "model": model,
                            "gold": gold,
                            "predicted_answer": predicted,
                            "generator_correct": actual_correctness,
                            "actual_correctness": actual_correctness,
                            "truncated": gen_truncated,
                            "generator_raw_output": gen_text,
                            "generator_finish_reason": gen_finish_reason,
                            "prompt_tokens": gen_resp.get("prompt_tokens"),
                            "completion_tokens": gen_resp.get("completion_tokens"),
                            "total_tokens_used": gen_resp.get("total_tokens"),
                            "generator_completion_avg_logprob": gen_resp.get("completion_avg_logprob"),
                            "generator_max_tokens": generator_max_tokens,
                            "generator_reasoning_budget": generator_reasoning_budget,
                            "verifier_prompt_tokens": ver_resp.get("prompt_tokens"),
                            "verifier_completion_tokens": ver_resp.get("completion_tokens"),
                            "verifier_total_tokens_used": ver_resp.get("total_tokens"),
                            "verifier_completion_avg_logprob": ver_resp.get("completion_avg_logprob"),
                            "verifier_max_tokens": verifier_max_tokens,
                            # Extra fields used by existing pipeline + verifier analysis
                            "generator_reasoning": gen_reasoning,
                            "verifier_raw_output": ver_text,
                            **verdict_prob_metrics,
                            "verifier_finish_reason": ver_finish_reason,
                            "verifier_truncated": ver_truncated,
                            "verifier_decision": ver_decision,
                            "verifier_correct": verifier_correct,
                            "verifier_lazy_accept": (ver_decision is True and actual_correctness is False),
                            "verifier_false_reject": (ver_decision is False and actual_correctness is True),
                            "verifier_abstained": ver_decision is None,
                            "budget": {
                                "pilot_config": pilot_config_name,
                                "gen_config": gen_config_name,
                                "total_max_tokens": total_max_tokens,
                                "generator_ratio": float(ratio),
                                "generator_max_tokens": generator_max_tokens,
                                "verifier_max_tokens": verifier_max_tokens,
                                "verifier_min_tokens": verifier_min_tokens,
                                "generator_reasoning_budget": generator_reasoning_budget,
                            },
                            "usage": {
                                "generator": {
                                    "prompt_tokens": gen_resp.get("prompt_tokens"),
                                    "completion_tokens": gen_resp.get("completion_tokens"),
                                    "total_tokens": gen_resp.get("total_tokens"),
                                    "completion_avg_logprob": gen_resp.get("completion_avg_logprob"),
                                    "completion_logprob_sum": gen_resp.get("completion_logprob_sum"),
                                    "completion_logprob_count": gen_resp.get("completion_logprob_count"),
                                    "model": gen_resp.get("model"),
                                    "finish_reason": gen_finish_reason,
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
                                    "finish_reason": ver_finish_reason,
                                    "response_id": ver_resp.get("response_id"),
                                },
                            },
                        }

                    except Exception as exc:
                        result = {
                            "id": item.get("id"),
                            "dataset": dataset,
                            "pilot_config": pilot_config_name,
                            "config": gen_config_name,
                            "model": model,
                            "error": repr(exc),
                            "budget": {
                                "pilot_config": pilot_config_name,
                                "gen_config": gen_config_name,
                                "total_max_tokens": total_max_tokens,
                                "generator_ratio": float(ratio),
                                "generator_max_tokens": generator_max_tokens,
                                "verifier_max_tokens": verifier_max_tokens,
                                "verifier_min_tokens": verifier_min_tokens,
                                "generator_reasoning_budget": generator_reasoning_budget,
                            },
                        }

                    results.append(result)

                    if save_every and (len(results) % save_every == 0):
                        save_jsonl(results, out_path)

            else:

                async def _run_ratio_async() -> list[dict[str, Any]]:
                    sem = asyncio.Semaphore(concurrency)
                    out: list[dict[str, Any]] = []
                    items: list[tuple[int, dict[str, Any]]] = []

                    for idx, item in enumerate(load_jsonl(input_path)):
                        if limit is not None and idx >= limit:
                            break
                        items.append((idx, item))

                    async def _process_one(item: dict[str, Any]) -> dict[str, Any]:
                        try:
                            gen_prompt = prompt_builder(item, reasoning_budget=generator_reasoning_budget)
                            async with sem:
                                gen_resp = await run_model_async(
                                    prompt=gen_prompt,
                                    model=model,
                                    temperature=temperature,
                                    max_tokens=generator_max_tokens,
                                    logprobs=True,
                                )

                            gen_text = gen_resp["text"]
                            gen_finish_reason = gen_resp.get("finish_reason")
                            gen_truncated = gen_finish_reason == "length"
                            gen_reasoning, predicted = parse_generation(gen_text)

                            actual_correctness = evaluator(predicted, item)

                            ver_prompt = build_verifier_prompt(
                                question=item.get("question", ""),
                                reasoning=gen_reasoning or "",
                                answer=predicted,
                            )

                            async with sem:
                                ver_resp = await run_model_async(
                                    prompt=ver_prompt,
                                    model=model,
                                    temperature=temperature,
                                    max_tokens=verifier_max_tokens,
                                    logprobs=True,
                                    top_logprobs=50,
                                    return_token_logprobs=True,
                                    token_logprobs_last_n=12,
                                )

                            ver_text = ver_resp["text"]
                            ver_finish_reason = ver_resp.get("finish_reason")
                            ver_truncated = ver_finish_reason == "length"
                            ver_decision = parse_verifier(ver_text)

                            chosen_verdict = "CORRECT" if ver_decision is True else "INCORRECT" if ver_decision is False else None
                            verdict_prob_metrics = _compute_verdict_probability_metrics(
                                ver_resp.get("token_logprobs"),
                                chosen_verdict=chosen_verdict,
                            )

                            verifier_correct = ver_decision is not None and ver_decision == actual_correctness
                            gold = item.get("answer", {}).get("ideal")

                            return {
                                "id": item.get("id"),
                                "dataset": dataset,
                                "pilot_config": pilot_config_name,
                                "config": gen_config_name,
                                "model": model,
                                "gold": gold,
                                "predicted_answer": predicted,
                                "generator_correct": actual_correctness,
                                "actual_correctness": actual_correctness,
                                "truncated": gen_truncated,
                                "generator_raw_output": gen_text,
                                "generator_finish_reason": gen_finish_reason,
                                "prompt_tokens": gen_resp.get("prompt_tokens"),
                                "completion_tokens": gen_resp.get("completion_tokens"),
                                "total_tokens_used": gen_resp.get("total_tokens"),
                                "generator_completion_avg_logprob": gen_resp.get("completion_avg_logprob"),
                                "generator_max_tokens": generator_max_tokens,
                                "generator_reasoning_budget": generator_reasoning_budget,
                                "verifier_prompt_tokens": ver_resp.get("prompt_tokens"),
                                "verifier_completion_tokens": ver_resp.get("completion_tokens"),
                                "verifier_total_tokens_used": ver_resp.get("total_tokens"),
                                "verifier_completion_avg_logprob": ver_resp.get("completion_avg_logprob"),
                                "verifier_max_tokens": verifier_max_tokens,
                                "generator_reasoning": gen_reasoning,
                                "verifier_raw_output": ver_text,
                                **verdict_prob_metrics,
                                "verifier_finish_reason": ver_finish_reason,
                                "verifier_truncated": ver_truncated,
                                "verifier_decision": ver_decision,
                                "verifier_correct": verifier_correct,
                                "verifier_lazy_accept": (ver_decision is True and actual_correctness is False),
                                "verifier_false_reject": (ver_decision is False and actual_correctness is True),
                                "verifier_abstained": ver_decision is None,
                                "budget": {
                                    "pilot_config": pilot_config_name,
                                    "gen_config": gen_config_name,
                                    "total_max_tokens": total_max_tokens,
                                    "generator_ratio": float(ratio),
                                    "generator_max_tokens": generator_max_tokens,
                                    "verifier_max_tokens": verifier_max_tokens,
                                    "verifier_min_tokens": verifier_min_tokens,
                                    "generator_reasoning_budget": generator_reasoning_budget,
                                },
                                "usage": {
                                    "generator": {
                                        "prompt_tokens": gen_resp.get("prompt_tokens"),
                                        "completion_tokens": gen_resp.get("completion_tokens"),
                                        "total_tokens": gen_resp.get("total_tokens"),
                                        "completion_avg_logprob": gen_resp.get("completion_avg_logprob"),
                                        "completion_logprob_sum": gen_resp.get("completion_logprob_sum"),
                                        "completion_logprob_count": gen_resp.get("completion_logprob_count"),
                                        "model": gen_resp.get("model"),
                                        "finish_reason": gen_finish_reason,
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
                                        "finish_reason": ver_finish_reason,
                                        "response_id": ver_resp.get("response_id"),
                                    },
                                },
                            }
                        except Exception as exc:
                            return {
                                "id": item.get("id"),
                                "dataset": dataset,
                                "pilot_config": pilot_config_name,
                                "config": gen_config_name,
                                "model": model,
                                "error": repr(exc),
                                "budget": {
                                    "pilot_config": pilot_config_name,
                                    "gen_config": gen_config_name,
                                    "total_max_tokens": total_max_tokens,
                                    "generator_ratio": float(ratio),
                                    "generator_max_tokens": generator_max_tokens,
                                    "verifier_max_tokens": verifier_max_tokens,
                                    "verifier_min_tokens": verifier_min_tokens,
                                    "generator_reasoning_budget": generator_reasoning_budget,
                                },
                            }

                    for batch in _iter_batches(items):
                        tasks = [asyncio.create_task(_process_one(item)) for _, item in batch]
                        batch_rows = await asyncio.gather(*tasks)
                        out.extend(batch_rows)
                        if save_every and (len(out) % save_every == 0):
                            save_jsonl(out, out_path)

                    return out

                results = asyncio.run(_run_ratio_async())

            save_jsonl(results, out_path)

            valid = [r for r in results if "error" not in r]

            # Smoke-check style completeness + truncation metrics (quick exploration)
            total_valid = len(valid)
            n_truncated = sum(1 for r in valid if r.get("truncated") is True)
            n_has_answer = sum(1 for r in valid if r.get("predicted_answer") is not None)
            n_complete = sum(
                1
                for r in valid
                if (r.get("truncated") is not True) and (r.get("predicted_answer") is not None)
            )
            n_verifier_truncated = sum(1 for r in valid if r.get("verifier_truncated") is True)

            verifier_correct_n = sum(1 for r in valid if r.get("verifier_decision") is True)
            verifier_incorrect_n = sum(1 for r in valid if r.get("verifier_decision") is False)
            verifier_abstain_n = sum(1 for r in valid if r.get("verifier_decision") is None)
            verifier_outputs_both_labels = verifier_correct_n > 0 and verifier_incorrect_n > 0

            # Confusion-matrix style counts, excluding abstentions where required by the thesis formulas.
            decided = [r for r in valid if r.get("verifier_decision") is not None]
            total_decided = len(decided)

            correct_accepts = sum(
                1
                for r in decided
                if r.get("actual_correctness") is True and r.get("verifier_decision") is True
            )
            false_reject_count = sum(
                1
                for r in decided
                if r.get("actual_correctness") is True and r.get("verifier_decision") is False
            )
            correct_rejections = sum(
                1
                for r in decided
                if r.get("actual_correctness") is False and r.get("verifier_decision") is False
            )
            lazy_accept_count = sum(
                1
                for r in decided
                if r.get("actual_correctness") is False and r.get("verifier_decision") is True
            )

            fpr_denom = lazy_accept_count + correct_rejections
            false_positive_rate = (lazy_accept_count / fpr_denom) if fpr_denom else 0.0

            fnr_denom = false_reject_count + correct_accepts
            false_negative_rate = (false_reject_count / fnr_denom) if fnr_denom else 0.0

            system_level_accuracy = ((correct_accepts + correct_rejections) / total_decided) if total_decided else 0.0

            # "Effective acceptance" = system produced a decision (i.e., did NOT abstain).
            effective_acceptance_rate = (total_decided / total_valid) if total_valid else 0.0

            summary = {
                "dataset": dataset,
                "pilot_config": pilot_config_name,
                "gen_config": gen_config_name,
                "model": model,
                "temperature": temperature,
                "n_total": len(results),
                "n_valid": len(valid),
                # Smoke-check style metrics
                "trunc_rate": (n_truncated / total_valid) if total_valid else 0.0,
                "has_answer_rate": (n_has_answer / total_valid) if total_valid else 0.0,
                "complete_rate": (n_complete / total_valid) if total_valid else 0.0,
                "verifier_decisions": {
                    "CORRECT": int(verifier_correct_n),
                    "INCORRECT": int(verifier_incorrect_n),
                    "ABSTAIN": int(verifier_abstain_n),
                },
                "verifier_outputs_both_labels": bool(verifier_outputs_both_labels),
                # Consistent naming for downstream analysis
                "generator_truncation_rate": (n_truncated / total_valid) if total_valid else 0.0,
                "verifier_truncation_rate": (n_verifier_truncated / total_valid) if total_valid else 0.0,
                "generator_accuracy": (
                    sum(1 for r in valid if r.get("actual_correctness") is True) / len(valid)
                    if valid else 0.0
                ),
                "verifier_accuracy": (
                    sum(1 for r in valid if r.get("verifier_correct") is True) / len(valid)
                    if valid else 0.0
                ),
                "lazy_accept_rate": (
                    sum(1 for r in valid if r.get("verifier_lazy_accept") is True) / len(valid)
                    if valid else 0.0
                ),
                "abstain_rate": (
                    sum(1 for r in valid if r.get("verifier_abstained") is True) / len(valid)
                    if valid else 0.0
                ),
                "false_positive_rate": float(false_positive_rate),
                "false_negative_rate": float(false_negative_rate),
                "system_level_accuracy": float(system_level_accuracy),
                "effective_acceptance_rate": float(effective_acceptance_rate),
                "budget": {
                    "total_max_tokens": total_max_tokens,
                    "generator_ratio": float(ratio),
                    "generator_max_tokens": generator_max_tokens,
                    "verifier_max_tokens": verifier_max_tokens,
                    "generator_reasoning_budget": generator_reasoning_budget,
                },
                "output": out_path,
            }

            summary_path = out_path.replace(".jsonl", "_summary.json")
            _save_json(summary, summary_path)

            created_outputs.append(
                {
                    "dataset": dataset,
                    "ratio": f"{float(ratio):.6f}",
                    "jsonl": out_path,
                    "summary": summary_path,
                }
            )

            print(
                f"[{dataset}] ratio={ratio:.2f} saved {len(results)} rows → {out_path} "
                f"(verifier_acc={summary['verifier_accuracy']:.2%}, lazy_accept={summary['lazy_accept_rate']:.2%})"
            )

    if write_manifest:
        # Best-effort only; never fail the run because manifest writing failed.
        try:
            manifest_path = str(Path(output_dir) / "latest_run_manifest.json")
            _save_json(
                {
                    "created_at_unix": run_started_at,
                    "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_started_at)),
                    "output_dir": output_dir,
                    "datasets": datasets,
                    "pilot_config": pilot_config_name,
                    "gen_config": gen_config_name,
                    "model": model,
                    "temperature": temperature,
                    "limit": limit,
                    "total_max_tokens": total_max_tokens,
                    "generator_reasoning_budget": target_reasoning_budget,
                    "generator_ratios": [float(r) for r in generator_ratios],
                    "files": created_outputs,
                },
                manifest_path,
            )
            print(f"Wrote manifest: {manifest_path}")
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pilot Generator→Verifier experiments using registries.")

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "strategyqa", "truthfulqa"],
        help="Datasets to run (default: all pilot datasets)",
    )
    parser.add_argument(
        "--pilot-config",
        default="pilot_default",
        choices=sorted(PILOT_CONFIGS.keys()),
    )
    parser.add_argument(
        "--gen-config",
        default="gen_medium",
        choices=sorted(CONFIGS.keys()),
    )
    parser.add_argument(
        "--total-max-tokens",
        type=int,
        default=None,
        help="Override PILOT_CONFIGS[--pilot-config].total_max_tokens for this run.",
    )
    parser.add_argument(
        "--generator-reasoning-budget",
        type=int,
        default=None,
        help="Override CONFIGS[--gen-config].reasoning_budget for this run (still clamped to generator budget).",
    )
    parser.add_argument(
        "--generator-ratios",
        type=float,
        nargs="+",
        default=None,
        help="Override PILOT_CONFIGS[--pilot-config].generator_ratios for this run (e.g. --generator-ratios 0.9 0.75 0.5 0.25).",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="data/experiments/pilot")
    parser.add_argument("--save-every", type=int, default=25)

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent OpenAI requests (default: 1 = sequential).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="How many rows to schedule per asyncio batch (default: 5x concurrency).",
    )

    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write latest_run_manifest.json into --output-dir (off by default).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    t0 = time.perf_counter()
    args = _parse_args()
    run_pilot(
        datasets=args.datasets,
        pilot_config_name=args.pilot_config,
        gen_config_name=args.gen_config,
        model=args.model,
        temperature=args.temperature,
        limit=args.limit,
        output_dir=args.output_dir,
        save_every=args.save_every,
        total_max_tokens=args.total_max_tokens,
        generator_reasoning_budget=args.generator_reasoning_budget,
        generator_ratios=args.generator_ratios,
        write_manifest=bool(args.write_manifest),
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )
    elapsed_s = time.perf_counter() - t0
    hh, rem = divmod(int(elapsed_s), 3600)
    mm, ss = divmod(rem, 60)
    print(f"Total runtime: {hh:02d}:{mm:02d}:{ss:02d} ({elapsed_s:.2f}s)")
