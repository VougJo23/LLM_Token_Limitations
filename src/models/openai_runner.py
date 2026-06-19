import os
from typing import Any
from openai import OpenAI

try:
    from openai import AsyncOpenAI  # type: ignore
except Exception:
    AsyncOpenAI = None  # type: ignore[assignment]


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) if AsyncOpenAI is not None else None

_together_client = None
_together_async_client = None


def _get_together_client():
    global _together_client
    if _together_client is None:
        _together_client = OpenAI(
            api_key=os.getenv("TOGETHERAI_KEY"),
            base_url="https://api.together.ai/v1",
        )
    return _together_client


def _get_together_async_client():
    global _together_async_client
    if _together_async_client is None:
        _together_async_client = AsyncOpenAI(
            api_key=os.getenv("TOGETHERAI_KEY"),
            base_url="https://api.together.ai/v1",
        )
    return _together_async_client

DEFAULT_MAX_TOKENS = 120
MAX_TOP_LOGPROBS = 20


def _extract_completion_logprob_metrics(choice0: Any) -> tuple[float | None, float | None, int | None]:
    """Extract simple logprob metrics from a ChatCompletion choice.

    Supports OpenAI format (logprobs.content) and TogetherAI/Qwen flat format
    (logprobs.token_logprobs).
    """
    logprobs = getattr(choice0, "logprobs", None)
    if not logprobs:
        return None, None, None

    # Try OpenAI format: logprobs.content (list of Logprob objects)
    content = getattr(logprobs, "content", None)
    if not content:
        try:
            content = logprobs.get("content")  # type: ignore[attr-defined]
        except Exception:
            content = None

    if content:
        token_logprobs: list[float] = []
        for tok in content:
            lp = getattr(tok, "logprob", None)
            if lp is None:
                try:
                    lp = tok.get("logprob")
                except Exception:
                    lp = None
            if lp is not None:
                try:
                    token_logprobs.append(float(lp))
                except (TypeError, ValueError):
                    pass
        if token_logprobs:
            s = float(sum(token_logprobs))
            n = int(len(token_logprobs))
            return (s / n) if n else None, s, n

    # TogetherAI/Qwen flat format: logprobs.token_logprobs (list of floats)
    flat_lps = getattr(logprobs, "token_logprobs", None)
    if flat_lps is None:
        try:
            flat_lps = logprobs.get("token_logprobs")  # type: ignore[attr-defined]
        except Exception:
            flat_lps = None
    if isinstance(flat_lps, list) and flat_lps:
        vals = [float(v) for v in flat_lps if v is not None]
        if vals:
            s = float(sum(vals))
            n = int(len(vals))
            return (s / n) if n else None, s, n

    return None, None, None


def _extract_token_logprobs(choice0: Any, *, last_n: int | None) -> list[dict[str, Any]] | None:
    """Extract token-level logprobs/top_logprobs from a ChatCompletion choice."""
    logprobs = getattr(choice0, "logprobs", None)
    if not logprobs:
        return None

    # Try OpenAI format: logprobs.content (list of Logprob objects)
    content = getattr(logprobs, "content", None)
    if not content:
        try:
            content = logprobs.get("content")  # type: ignore[attr-defined]
        except Exception:
            content = None

    if content:
        toks = list(content)
        if isinstance(last_n, int) and last_n > 0:
            toks = toks[-last_n:]

        out: list[dict[str, Any]] = []
        for tok in toks:
            token = getattr(tok, "token", None)
            if token is None:
                try:
                    token = tok.get("token")
                except Exception:
                    token = None

            lp = getattr(tok, "logprob", None)
            if lp is None:
                try:
                    lp = tok.get("logprob")
                except Exception:
                    lp = None

            top = getattr(tok, "top_logprobs", None)
            if top is None:
                try:
                    top = tok.get("top_logprobs")
                except Exception:
                    top = None

            top_out: list[dict[str, Any]] | None = None
            if top:
                top_out = []
                for alt in list(top):
                    alt_tok = getattr(alt, "token", None)
                    if alt_tok is None:
                        try:
                            alt_tok = alt.get("token")
                        except Exception:
                            alt_tok = None

                    alt_lp = getattr(alt, "logprob", None)
                    if alt_lp is None:
                        try:
                            alt_lp = alt.get("logprob")
                        except Exception:
                            alt_lp = None

                    top_out.append(
                        {
                            "token": alt_tok,
                            "logprob": float(alt_lp) if alt_lp is not None else None,
                        }
                    )

            out.append(
                {
                    "token": token,
                    "logprob": float(lp) if lp is not None else None,
                    "top_logprobs": top_out,
                }
            )

        return out

    # TogetherAI/Qwen flat format: logprobs.tokens, .token_logprobs, .top_logprobs
    tokens = getattr(logprobs, "tokens", None)
    if tokens is None:
        try:
            tokens = logprobs.get("tokens")  # type: ignore[attr-defined]
        except Exception:
            tokens = None
    token_lps = getattr(logprobs, "token_logprobs", None)
    if token_lps is None:
        try:
            token_lps = logprobs.get("token_logprobs")
        except Exception:
            token_lps = None
    if not isinstance(tokens, list) or not isinstance(token_lps, list):
        return None

    toks = list(zip(tokens, token_lps))
    if isinstance(last_n, int) and last_n > 0:
        toks = toks[-last_n:]

    # TogetherAI flat top_logprobs: list of dicts {token: logprob, ...}
    flat_top = getattr(logprobs, "top_logprobs", None)
    if flat_top is None:
        try:
            flat_top = logprobs.get("top_logprobs")
        except Exception:
            flat_top = None
    if isinstance(flat_top, list) and len(flat_top) == len(tokens):
        offset = len(tokens) - len(toks)
        flat_top_slice = flat_top[offset:] if offset else flat_top
    else:
        flat_top_slice = [None] * len(toks)

    out = []
    for (tok, lp), top_d in zip(toks, flat_top_slice):
        top_out = None
        if isinstance(top_d, dict):
            top_out = [
                {"token": t, "logprob": float(v) if v is not None else None}
                for t, v in top_d.items()
            ]
        out.append({
            "token": tok,
            "logprob": float(lp) if lp is not None else None,
            "top_logprobs": top_out,
        })

    return out


async def run_model_async(
    prompt: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    return_token_logprobs: bool = False,
    token_logprobs_last_n: int | None = 12,
    *,
    provider: str = "openai",
):
    """Run a single-turn chat completion asynchronously."""
    together_providers = ("qwen", "llama")
    if provider == "openai":
        selected_client = async_client
        max_allowed_top_logprobs = MAX_TOP_LOGPROBS
    elif provider in together_providers:
        selected_client = _get_together_async_client()
        max_allowed_top_logprobs = 5
    else:
        raise ValueError(f"Unsupported provider: {provider!r}")

    if selected_client is None:
        raise RuntimeError("Selected AsyncOpenAI client is not available.")

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if logprobs:
        if provider in together_providers:
            tlp = int(top_logprobs) if top_logprobs is not None else 1
            if tlp > max_allowed_top_logprobs:
                tlp = max_allowed_top_logprobs
            kwargs["extra_body"] = {"logprobs": max(1, tlp)}
        else:
            kwargs["logprobs"] = True
            if top_logprobs is not None:
                tlp = int(top_logprobs)
                if tlp > max_allowed_top_logprobs:
                    tlp = max_allowed_top_logprobs
                if tlp > 0:
                    kwargs["top_logprobs"] = tlp

    try:
        response = await selected_client.chat.completions.create(**kwargs)
    except Exception as exc:
        if logprobs:
            msg = str(exc).lower()
            if "logprobs" in msg and ("unknown" in msg or "unsupported" in msg or "unrecognized" in msg):
                if provider in together_providers:
                    kwargs.pop("extra_body", None)
                else:
                    kwargs.pop("logprobs", None)
                    kwargs.pop("top_logprobs", None)
                response = await selected_client.chat.completions.create(**kwargs)
            else:
                raise
        else:
            raise

    choice0 = response.choices[0]
    text = choice0.message.content
    finish_reason = getattr(choice0, "finish_reason", None)
    usage = getattr(response, "usage", None)

    completion_avg_logprob, completion_logprob_sum, completion_logprob_count = _extract_completion_logprob_metrics(choice0)

    token_logprobs = (
        _extract_token_logprobs(choice0, last_n=token_logprobs_last_n)
        if (logprobs and return_token_logprobs)
        else None
    )

    return {
        "text": text,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "completion_avg_logprob": completion_avg_logprob,
        "completion_logprob_sum": completion_logprob_sum,
        "completion_logprob_count": completion_logprob_count,
        "token_logprobs": token_logprobs,
        "model": model,
        "finish_reason": finish_reason,
        "response_id": getattr(response, "id", None),
    }


async def run_model_text(
    prompt: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0,
    provider: str = "openai",
):
    result = await run_model_async(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        provider=provider,
    )
    return result["text"]
