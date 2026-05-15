CONFIGS = {
    "gen_light": {
        "max_tokens": 60,
        "reasoning_budget": 30
    },
    "gen_medium": {
        "max_tokens": 120,
        "reasoning_budget": 80
    },
    "gen_heavy": {
        "max_tokens": 200,
        "reasoning_budget": 150
    }
}


PILOT_CONFIGS = {
    "pilot_default": {
        # Total completion tokens budget across (generator + verifier)
        "total_max_tokens": 160,
        # Sweep generator share; verifier gets the remainder
        "generator_ratios": [0.9, 0.75, 0.5, 0.25],
        # Avoid accidentally starving the verifier
        "verifier_min_tokens": 8,
        # Safety: cap generator reasoning budget to <= generator max tokens
        "generator_reasoning_budget_scale": 0.8,
    }
}
