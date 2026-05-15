import re


_VERDICT_RE = re.compile(r"\b(INCORRECT|CORRECT)\b", re.IGNORECASE)


def parse_verifier(output: str):
    """Return True/False if a verdict is found, else None."""

    if not output:
        return None

    match = _VERDICT_RE.search(str(output).strip())
    if not match:
        return None

    return match.group(1).upper() == "CORRECT"
