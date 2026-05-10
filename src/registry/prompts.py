from src.prompts.gsm8k_prompt import build_gsm8k_prompt
from src.prompts.truthfulqa_prompt import build_truthfulqa_prompt
from src.prompts.strategyqa_prompt import build_strategyqa_prompt

PROMPT_REGISTRY = {
    "gsm8k": gsm8k_prompt,
    "truthfulqa": truthfulqa_prompt,
    "strategyqa": strategyqa_prompt,
}
