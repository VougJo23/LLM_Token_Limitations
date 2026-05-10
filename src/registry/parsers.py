from src.parsers.gsm8k import parse_gsm8k
from src.parsers.truthfulqa import parse_truthfulqa
from src.parsers.strategyqa import parse_strategyqa

PARSER_REGISTRY = {
    "gsm8k": parse_gsm8k,
    "truthfulqa": parse_truthfulqa,
    "strategyqa": parse_strategyqa,
}
