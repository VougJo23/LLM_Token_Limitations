CONFIGS = {

    # Based on Difficulty
    "easy": {
        "total_max_tokens": 128,
        "generator_ratios": [0.9, 0.75, 0.6, 0.45],
        "verifier_min_tokens": 40,
        "generator_reasoning_budget_scale": 0.8,
    },

    "medium": {
        "total_max_tokens": 192,
        "generator_ratios": [0.9, 0.75, 0.6, 0.45],
        "verifier_min_tokens": 50,
        "generator_reasoning_budget_scale": 0.8,
    },

    "hard": {
        "total_max_tokens": 256,
        "generator_ratios": [0.9, 0.75, 0.6, 0.45],
        "verifier_min_tokens": 50,
        "generator_reasoning_budget_scale": 0.8,
    },
}



CONFIGS2 = {
    "easy": {
        "total_max_tokens": 128,
        "verifier_ratios": [0.9, 0.75, 0.6, 0.45],
        "verifier_min_tokens": 40
    },

    "medium": {
        "total_max_tokens": 192,
        "verifier_ratios": [0.9, 0.75, 0.6, 0.45],
        "verifier_min_tokens": 50
    },

    "hard": {
        "total_max_tokens": 256,
        "verifier_ratios": [0.9, 0.75, 0.6, 0.45],
        "verifier_min_tokens": 50
    },
}

CONFIGS_UNLIMITED = {

    "easy": {
        "total_max_tokens": 400,
        "verifier_ratios": [0.9, 0.75, 0.6, 0.45],
        "verifier_min_tokens": 40
    },

    "medium": {
        "total_max_tokens": 500,
        "verifier_ratios": [0.9, 0.75, 0.6, 0.45],
        "verifier_min_tokens": 50
    },

    "hard": {
        "total_max_tokens": 600,
        "verifier_ratios": [0.9, 0.75, 0.6, 0.45],
        "verifier_min_tokens": 50
    },
}



# PILOT_CONFIGS = {
    
#     "pilot_default": {
#         "total_max_tokens": 300,
#         "generator_ratios": [0.9, 0.8, 0.7, 0.6],
#         "verifier_min_tokens": 10,
#         "generator_reasoning_budget_scale": 0.8,
#     },
    
#     # initial configuration
#     "pilot_default_initial": {
#         "total_max_tokens": 160,
#         "generator_ratios": [0.9, 0.75, 0.5, 0.25],
#         "verifier_min_tokens": 15,
#         "generator_reasoning_budget_scale": 0.8,
#     },
# }


# PILOT2_CONFIGS = {
#     "pilot2_default": {
#         "verifier_total_max_tokens": 160,
#         "verifier_ratios": [0.9, 0.8, 0.7, 0.6], #, 0.25
#         "verifier_min_tokens": 1,
#         "generator_reasoning_budget_scale": 0.8,
#     },
# }
