import re

#_VERDICT_LINE_RE = re.compile(r"\b(INCORRECT|CORRECT)\b", re.IGNORECASE)


def parse_verifier(output: str):
    if not output:
        return None

    text = str(output).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        upper = ln.upper()
        # Accept bare verdict line
        if upper in ("CORRECT", "INCORRECT"):
            return upper == "CORRECT"

        # Accept trailing punctuation
        cleaned = re.sub(r"[.!]+$", "", upper)
        if cleaned in ("CORRECT", "INCORRECT"):
            return cleaned == "CORRECT"

        # Accept "Verdict: CORRECT/INCORRECT" format
        m = re.search(r"VERDICT\s*:\s*(CORRECT|INCORRECT)", ln, re.IGNORECASE)
        if m:
            return m.group(1).upper() == "CORRECT"

    return None
