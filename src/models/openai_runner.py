import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_MAX_TOKENS = 120


def run_model(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0,
):
    """Run a single-turn chat completion.

    Returns a dict so experiments can log token usage reliably.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    choice0 = response.choices[0]
    text = choice0.message.content
    finish_reason = getattr(choice0, "finish_reason", None)
    usage = getattr(response, "usage", None)

    return {
        "text": text,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "model": model,
        "finish_reason": finish_reason,
        "response_id": getattr(response, "id", None),
    }


def run_model_text(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0,
):
    """Backwards-compatible helper for callers that only want the text."""

    return run_model(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )["text"]
