CONFIGS = {
    "gen_light": {
        "max_tokens": 140,
        "reasoning_budget": 100
    },
    "gen_medium": {
        "max_tokens": 220,
        "reasoning_budget": 160
    },
    "gen_heavy": {
        "max_tokens": 320,
        "reasoning_budget": 240
    }
}


PILOT_CONFIGS = {
    "pilot_default": {
        
        "total_max_tokens": 220,

        "generator_ratios": [0.9, 0.8, 0.7, 0.6],
        # Avoid accidentally starving the verifier
        "verifier_min_tokens": 10,
        # Safety: cap generator reasoning budget to <= generator max tokens
        "generator_reasoning_budget_scale": 0.8,
    },
    # initial configuration
    "pilot_default_initial": {
        "total_max_tokens": 160,
        "generator_ratios": [0.9, 0.75, 0.5, 0.25],
        "verifier_min_tokens": 15,
        "generator_reasoning_budget_scale": 0.8,
    },
}


# Pilot2: generator is run once per question (fixed budget);
# verifier is swept over multiple token budgets to study truncation/FP/FN.
PILOT2_CONFIGS = {
    "pilot2_default": {
        # Maximum verifier budget; per-run verifier budgets are computed as:
        # verifier_max_tokens = max(verifier_min_tokens, int(verifier_total_max_tokens * verifier_ratio))
        "verifier_total_max_tokens": 160,
        "verifier_ratios": [0.9, 0.8, 0.7, 0.6], #, 0.25
        "verifier_min_tokens": 1,
        # Prompt-level reasoning budget is clamped to <= generator_max_tokens * scale
        "generator_reasoning_budget_scale": 0.8,
    },
}
