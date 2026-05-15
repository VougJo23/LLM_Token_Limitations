from src.parsers.gsm8k import parse_gsm8k
from src.parsers.gsm8k import parse_gsm8k_generation
from src.parsers.strategyqa import parse_strategyqa, parse_strategyqa_generation
from src.parsers.truthfulqa import parse_truthfulqa, parse_truthfulqa_generation

ANSWER_PARSER_REGISTRY = {
    "gsm8k": parse_gsm8k,
    "truthfulqa": parse_truthfulqa,
    "strategyqa": parse_strategyqa,
}

GENERATION_PARSER_REGISTRY = {
    "gsm8k": parse_gsm8k_generation,
    "truthfulqa": parse_truthfulqa_generation,
    "strategyqa": parse_strategyqa_generation,
}

# Backwards-compatible alias
PARSER_REGISTRY = ANSWER_PARSER_REGISTRY
