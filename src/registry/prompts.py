from src.prompts.gsm8k import gsm8k_generator_prompt
from src.prompts.strategyqa import strategyqa_generator_prompt
from src.prompts.truthfulqa import truthfulqa_generator_prompt


GENERATOR_PROMPT_REGISTRY = {
    "gsm8k": gsm8k_generator_prompt,
    "strategyqa": strategyqa_generator_prompt,
    "truthfulqa": truthfulqa_generator_prompt,
}

# Backwards-compatible alias
PROMPT_REGISTRY = GENERATOR_PROMPT_REGISTRY
