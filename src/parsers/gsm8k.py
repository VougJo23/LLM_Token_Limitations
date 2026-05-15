import re


def parse_gsm8k_generation(output):

    reasoning = None
    answer = None

    reasoning_match = re.search(
        r"Reasoning:(.*?)(Answer:|$)",
        output,
        re.DOTALL
    )

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    answer_match = re.search(
        r"Answer:\s*(-?\d+\.?\d*)",
        output
    )

    if answer_match:
        answer = float(answer_match.group(1))

    return reasoning, answer


def parse_gsm8k(text):
    """Extract a single numeric answer from a model output.

    Accepts raw strings or a dict that contains a `text` field.
    """

    if isinstance(text, dict):
        text = text.get("text") or text.get("output")

    if text is None:
        return None

    match = re.search(r"Answer:\s*(-?\d+\.?\d*)", str(text))
    return float(match.group(1)) if match else None
