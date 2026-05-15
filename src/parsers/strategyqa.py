import re
from typing import Optional, Tuple


_BOOL_RE = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)


def _to_bool(token: str) -> Optional[bool]:
    token = token.strip().lower()
    if token in {"true", "yes"}:
        return True
    if token in {"false", "no"}:
        return False
    return None


def parse_strategyqa(output: str) -> Optional[bool]:
    if not output:
        return None

    # Prefer an explicit Answer: line if present
    answer_line = re.search(r"Answer:\s*(.*)$", output, re.IGNORECASE | re.MULTILINE)
    if answer_line:
        match = _BOOL_RE.search(answer_line.group(1))
        if match:
            return _to_bool(match.group(1))

    match = _BOOL_RE.search(output)
    if match:
        return _to_bool(match.group(1))

    return None


def parse_strategyqa_generation(output: str) -> Tuple[Optional[str], Optional[bool]]:
    if not output:
        return None, None

    reasoning = None
    reasoning_match = re.search(r"Reasoning:(.*?)(Answer:|$)", output, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return reasoning, parse_strategyqa(output)
