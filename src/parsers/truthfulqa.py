import re
from typing import Optional, Tuple


def parse_truthfulqa(output: str) -> Optional[str]:
    if output is None:
        return None

    text = str(output).strip()
    if not text:
        return None

    answer_line = re.search(r"Answer:\s*(.*)$", text, re.IGNORECASE | re.MULTILINE)
    if answer_line:
        return answer_line.group(1).strip() or None

    # Fallback: use full text (pilot-friendly)
    return text


def parse_truthfulqa_generation(output: str) -> Tuple[Optional[str], Optional[str]]:
    if not output:
        return None, None

    reasoning = None
    reasoning_match = re.search(r"Reasoning:(.*?)(Answer:|$)", output, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    answer = parse_truthfulqa(output)
    return reasoning, answer
