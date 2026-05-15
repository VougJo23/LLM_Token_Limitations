from __future__ import annotations

from typing import Any

from src.models.openai_runner import run_model
from src.parsers.gsm8k import parse_gsm8k_generation
from src.parsers.verifier import parse_verifier
from src.prompts.gsm8k import gsm8k_generator_prompt
from src.prompts.verifier import build_verifier_prompt
from src.utils.io import load_jsonl, save_jsonl


def _numeric_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def _allocate_completion_budgets(total_max_tokens: int, generator_ratio: float) -> tuple[int, int]:
    if total_max_tokens <= 1:
        return 1, 1
    generator_max = max(1, int(total_max_tokens * generator_ratio))
    verifier_max = max(1, total_max_tokens - generator_max)
    return generator_max, verifier_max


def run_pilot_verifier(
    input_path: str = "data/pilot/gsm8k_pilot.jsonl",
    output_path: str = "data/experiments/pilot/gsm8k_verifier_results.jsonl",
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    total_max_tokens: int = 160,
    generator_ratio: float = 0.75,
    generator_reasoning_budget: int | None = None,
    limit: int | None = None,
    save_every: int = 25,
):
    """Run a lightweight pilot: Generator → Verifier.

    `total_max_tokens` is split across the two calls using `generator_ratio`.
    """

    if not (0.0 < generator_ratio < 1.0):
        raise ValueError("generator_ratio must be between 0 and 1 (exclusive)")

    generator_max_tokens, verifier_max_tokens = _allocate_completion_budgets(
        total_max_tokens=total_max_tokens,
        generator_ratio=generator_ratio,
    )

    if generator_reasoning_budget is None:
        generator_reasoning_budget = max(1, int(generator_max_tokens * 0.8))

    results: list[dict[str, Any]] = []

    for idx, item in enumerate(load_jsonl(input_path)):
        if limit is not None and idx >= limit:
            break

        try:
            generator_prompt = gsm8k_generator_prompt(
                item,
                reasoning_budget=generator_reasoning_budget,
            )

            gen_resp = run_model(
                prompt=generator_prompt,
                model=model,
                temperature=temperature,
                max_tokens=generator_max_tokens,
            )

            generator_raw_output = gen_resp["text"]
            gen_finish_reason = gen_resp.get("finish_reason")
            gen_truncated = gen_finish_reason == "length"
            generator_reasoning, predicted_answer = parse_gsm8k_generation(generator_raw_output)

            gold = item["answer"]["ideal"]
            actual_correctness = _numeric_equal(predicted_answer, gold)

            verifier_prompt = build_verifier_prompt(
                question=item["question"],
                reasoning=generator_reasoning or "",
                answer=predicted_answer,
            )

            ver_resp = run_model(
                prompt=verifier_prompt,
                model=model,
                temperature=temperature,
                max_tokens=verifier_max_tokens,
            )

            verifier_raw_output = ver_resp["text"]
            ver_finish_reason = ver_resp.get("finish_reason")
            ver_truncated = ver_finish_reason == "length"
            verifier_decision = parse_verifier(verifier_raw_output)
            verifier_correct = (
                verifier_decision is not None
                and verifier_decision == actual_correctness
            )

            result: dict[str, Any] = {
                "id": item.get("id"),
                "dataset": "gsm8k",
                "config": "pilot_verifier",
                "model": model,
                "question": item.get("question"),
                "gold": gold,
                "predicted_answer": predicted_answer,
                "generator_correct": actual_correctness,
                "actual_correctness": actual_correctness,
                "truncated": gen_truncated,
                "generator_raw_output": generator_raw_output,
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

                "generator_reasoning": generator_reasoning,
                "verifier_raw_output": verifier_raw_output,
                "verifier_finish_reason": ver_finish_reason,
                "verifier_truncated": ver_truncated,
                "verifier_decision": verifier_decision,
                "verifier_correct": verifier_correct,
                "verifier_lazy_accept": (verifier_decision is True and actual_correctness is False),
                "verifier_false_reject": (verifier_decision is False and actual_correctness is True),
                "verifier_abstained": verifier_decision is None,
                "budget": {
                    "total_max_tokens": total_max_tokens,
                    "generator_ratio": generator_ratio,
                    "generator_max_tokens": generator_max_tokens,
                    "verifier_max_tokens": verifier_max_tokens,
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
                "question": item.get("question"),
                "error": repr(exc),
                "budget": {
                    "total_max_tokens": total_max_tokens,
                    "generator_ratio": generator_ratio,
                    "generator_max_tokens": generator_max_tokens,
                    "verifier_max_tokens": verifier_max_tokens,
                    "generator_reasoning_budget": generator_reasoning_budget,
                },
            }

        results.append(result)

        if save_every and (len(results) % save_every == 0):
            save_jsonl(results, output_path)

    save_jsonl(results, output_path)
    print(f"Saved {len(results)} results → {output_path}")
    return results


if __name__ == "__main__":
    run_pilot_verifier()
