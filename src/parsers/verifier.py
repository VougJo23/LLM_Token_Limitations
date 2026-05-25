import re


_VERDICT_RE = re.compile(r"\b(INCORRECT|CORRECT)\b", re.IGNORECASE)


def parse_verifier(output: str):
    """Parse the verifier verdict.

    Returns:
    - True  => CORRECT
    - False => INCORRECT
    - None  => no valid standalone verdict line found

    Notes:
    Verifier responses often contain words like "correct" in their justification.
    We intentionally only accept a standalone final verdict line to avoid false
    positives (e.g. "mathematically correct" while the final verdict is INCORRECT).
    """

    if not output:
        return None

    text = str(output).strip()

    # Prefer a standalone verdict line, scanning from the bottom.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        upper = ln.upper()
        if upper in ("CORRECT", "INCORRECT"):
            return upper == "CORRECT"

        # Be slightly tolerant of trivial trailing punctuation.
        upper_clean = re.sub(r"[.!]+$", "", upper)
        if upper_clean in ("CORRECT", "INCORRECT"):
            return upper_clean == "CORRECT"

    # Fallback: if the output is a single token somewhere, accept it.
    # (This keeps behavior reasonable if the verifier truly outputs only CORRECT/INCORRECT.)
    match = _VERDICT_RE.fullmatch(text)
    if match:
        return match.group(1).upper() == "CORRECT"

    return None
