import re

def parse_gsm8k(output: str):
    if not output:
        return None

    matches = re.findall(r"-?\d+\.?\d*", output)
    return float(matches[-1]) if matches else None
