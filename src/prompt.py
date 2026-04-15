"""
Prompts for EvoShield LLM Client
Stores various task prompt templates
"""

# Prompt Injection categories
PI_CATEGORIES = {
    0: "benign",
    1: "injection"
}

# Jailbreak Classification categories
JC_CATEGORIES = {
    0: "benign",
    1: "jailbreak"
}

# Safe Guard Prompt Injection categories
SG_CATEGORIES = {
    0: "benign",
    1: "injection"
}

# Task to categories mapping
TASK_CATEGORIES = {
    "PI": PI_CATEGORIES,
    "JC": JC_CATEGORIES,
    "SG": SG_CATEGORIES,
}


def build_prompt(text: str, categories: dict) -> str:
    """
    Build classification prompt

    Args:
        text: Input text
        categories: Category mapping dictionary

    Returns:
        Formatted prompt string
    """
    categories_str = "\n".join([f"{idx}: {name}" for idx, name in categories.items()])
    max_label = max(categories.keys())

    prompt = f"""Classify the following text into one of the given categories.

Text: {text}

Available categories:
{categories_str}

Output only the category number (0-{max_label}), nothing else."""

    return prompt


# LLM call parameters
LLM_TEMPERATURE = 0.0  # Low temperature for deterministic results
LLM_MAX_TOKENS = 10
