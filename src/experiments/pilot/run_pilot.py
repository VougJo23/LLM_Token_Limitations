from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models.openai_runner import run_model
from src.parsers.verifier import parse_verifier
from src.prompts.verifier import build_verifier_prompt
from src.registry.configs import CONFIGS, PILOT_CONFIGS
from src.registry.datasets import DATASET_PATHS
from src.registry.evaluators import EVALUATOR_REGISTRY
from src.registry.parsers import GENERATION_PARSER_REGISTRY
from src.registry.prompts import GENERATOR_PROMPT_REGISTRY
from src.utils.io import load_jsonl, save_jsonl


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
) -> None:
    pilot_cfg = PILOT_CONFIGS[pilot_config_name]
    gen_cfg = CONFIGS[gen_config_name]

    total_max_tokens = int(pilot_cfg["total_max_tokens"])
    generator_ratios = list(pilot_cfg["generator_ratios"])
    verifier_min_tokens = int(pilot_cfg.get("verifier_min_tokens", 1))
    reasoning_budget_scale = float(pilot_cfg.get("generator_reasoning_budget_scale", 0.8))

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

            # Prompt-level reasoning budget (soft). Clamp to what the completion budget can actually support.
            generator_reasoning_budget = min(
                int(gen_cfg["reasoning_budget"]),
                max(1, int(generator_max_tokens * reasoning_budget_scale)),
            )

            results: list[dict[str, Any]] = []

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
                    )

                    ver_text = ver_resp["text"]
                    ver_finish_reason = ver_resp.get("finish_reason")
                    ver_truncated = ver_finish_reason == "length"
                    ver_decision = parse_verifier(ver_text)

                    verifier_correct = (
                        ver_decision is not None
                        and ver_decision == actual_correctness
                    )

                    gold = item.get("answer", {}).get("ideal")

                    # Canonical/flat fields (easy to analyze in pandas)
                    result = {
                        "id": item.get("id"),
                        "dataset": dataset,
                        "pilot_config": pilot_config_name,
                        "config": gen_config_name,
                        "model": model,
                        "question": item.get("question"),
                        "gold": gold,
                        "predicted_answer": predicted,
                        "generator_correct": actual_correctness,
                        # Back-compat alias (older analysis code uses this key)
                        "actual_correctness": actual_correctness,
                        "truncated": gen_truncated,
                        "generator_raw_output": gen_text,
                        "generator_finish_reason": gen_finish_reason,
                        "prompt_tokens": gen_resp.get("prompt_tokens"),
                        "completion_tokens": gen_resp.get("completion_tokens"),
                        "total_tokens_used": gen_resp.get("total_tokens"),
                        "generator_max_tokens": generator_max_tokens,
                        "generator_reasoning_budget": generator_reasoning_budget,

                        "verifier_prompt_tokens": ver_resp.get("prompt_tokens"),
                        "verifier_completion_tokens": ver_resp.get("completion_tokens"),
                        "verifier_total_tokens_used": ver_resp.get("total_tokens"),
                        "verifier_max_tokens": verifier_max_tokens,

                        # Extra fields used by existing pipeline + verifier analysis
                        "generator_reasoning": gen_reasoning,
                        "verifier_raw_output": ver_text,
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
                                "model": gen_resp.get("model"),
                                "finish_reason": gen_finish_reason,
                                "response_id": gen_resp.get("response_id"),
                            },
                            "verifier": {
                                "prompt_tokens": ver_resp.get("prompt_tokens"),
                                "completion_tokens": ver_resp.get("completion_tokens"),
                                "total_tokens": ver_resp.get("total_tokens"),
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
                        "question": item.get("question"),
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
                    out_path = f"{output_dir}/{dataset}_{pilot_config_name}_{gen_config_name}_r{int(float(ratio) * 100)}.jsonl"
                    save_jsonl(results, out_path)

            out_path = f"{output_dir}/{dataset}_{pilot_config_name}_{gen_config_name}_r{int(float(ratio) * 100)}.jsonl"
            save_jsonl(results, out_path)

            valid = [r for r in results if "error" not in r]
            summary = {
                "dataset": dataset,
                "pilot_config": pilot_config_name,
                "gen_config": gen_config_name,
                "model": model,
                "temperature": temperature,
                "n_total": len(results),
                "n_valid": len(valid),
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

            print(
                f"[{dataset}] ratio={ratio:.2f} saved {len(results)} rows → {out_path} "
                f"(verifier_acc={summary['verifier_accuracy']:.2%}, lazy_accept={summary['lazy_accept_rate']:.2%})"
            )


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
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="data/experiments/pilot")
    parser.add_argument("--save-every", type=int, default=25)

    return parser.parse_args()


if __name__ == "__main__":
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
    )
