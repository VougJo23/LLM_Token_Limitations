from __future__ import annotations

import re
from typing import Any, Callable


def _numeric_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def _norm_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def eval_gsm8k(pred: Any, item: dict) -> bool:
    gold = item.get("answer", {}).get("ideal")
    return _numeric_equal(pred, gold)


def eval_strategyqa(pred: Any, item: dict) -> bool:
    gold = item.get("answer", {}).get("ideal")
    return pred is not None and pred == gold


def eval_truthfulqa(pred: Any, item: dict) -> bool:
    if pred is None:
        return False

    pred_n = _norm_text(pred)

    answer = item.get("answer", {})
    ideal = _norm_text(answer.get("ideal"))
    alternatives = [_norm_text(x) for x in (answer.get("alternatives") or [])]
    incorrect = [_norm_text(x) for x in (answer.get("incorrect") or [])]

    if pred_n in incorrect:
        return False

    if pred_n == ideal:
        return True

    return pred_n in alternatives


EVALUATOR_REGISTRY: dict[str, Callable[[Any, dict], bool]] = {
    "gsm8k": eval_gsm8k,
    "strategyqa": eval_strategyqa,
    "truthfulqa": eval_truthfulqa,
}
