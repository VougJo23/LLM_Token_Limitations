import os
from typing import Any
from openai import OpenAI

try:
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) if AsyncOpenAI is not None else None

DEFAULT_MAX_TOKENS = 120
MAX_TOP_LOGPROBS = 20


def _extract_completion_logprob_metrics(choice0: Any) -> tuple[float | None, float | None, int | None]:
    """Extract simple logprob metrics from a ChatCompletion choice.

    Returns (avg_logprob, sum_logprob, n_tokens) for completion tokens if available.
    """

    logprobs = getattr(choice0, "logprobs", None)
    if not logprobs:
        return None, None, None

    content = getattr(logprobs, "content", None)
    if not content:
        # Some SDK versions may represent logprobs as dict-like.
        try:
            content = logprobs.get("content")  # type: ignore[attr-defined]
        except Exception:
            content = None

    if not content:
        return None, None, None

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

    if not token_logprobs:
        return None, None, None

    s = float(sum(token_logprobs))
    n = int(len(token_logprobs))
    return (s / n) if n else None, s, n


def _extract_token_logprobs(choice0: Any, *, last_n: int | None) -> list[dict[str, Any]] | None:
    """Extract token-level logprobs/top_logprobs from a ChatCompletion choice.

    Returns a list of dicts like:
      {"token": str, "logprob": float|None, "top_logprobs": [{"token": str, "logprob": float|None}, ...]}

    If last_n is provided, returns only the last N tokens.
    """

    logprobs = getattr(choice0, "logprobs", None)
    if not logprobs:
        return None

    content = getattr(logprobs, "content", None)
    if not content:
        try:
            content = logprobs.get("content")  # type: ignore[attr-defined]
        except Exception:
            content = None

    if not content:
        return None

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


def run_model(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    return_token_logprobs: bool = False,
    token_logprobs_last_n: int | None = 12,
):
    """Run a single-turn chat completion.
    Returns a dict so experiments can log token usage reliably.
    """

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if logprobs:
        kwargs["logprobs"] = True
        if top_logprobs is not None:
            tlp = int(top_logprobs)
            if tlp > MAX_TOP_LOGPROBS:
                tlp = MAX_TOP_LOGPROBS
            if tlp > 0:
                kwargs["top_logprobs"] = tlp

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        
        if logprobs:
            msg = str(exc).lower()
            if "logprobs" in msg and ("unknown" in msg or "unsupported" in msg or "unrecognized" in msg):
                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)
                response = client.chat.completions.create(**kwargs)
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


async def run_model_async(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    return_token_logprobs: bool = False,
    token_logprobs_last_n: int | None = 12,
):
    """Async variant of run_model (uses AsyncOpenAI)."""

    if async_client is None:
        raise RuntimeError(
            "AsyncOpenAI is not available in this environment. "
            "Upgrade the `openai` package to a version that provides AsyncOpenAI, "
            "or run with concurrency=1."
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if logprobs:
        kwargs["logprobs"] = True
        if top_logprobs is not None:
            tlp = int(top_logprobs)
            if tlp > MAX_TOP_LOGPROBS:
                tlp = MAX_TOP_LOGPROBS
            if tlp > 0:
                kwargs["top_logprobs"] = tlp

    try:
        response = await async_client.chat.completions.create(**kwargs)
    except Exception as exc:

        if logprobs:
            msg = str(exc).lower()
            if "logprobs" in msg and (
                "unknown" in msg or "unsupported" in msg or "unrecognized" in msg
            ):
                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)
                response = await async_client.chat.completions.create(**kwargs)
            else:
                raise
        else:
            raise

    choice0 = response.choices[0]
    text = choice0.message.content
    finish_reason = getattr(choice0, "finish_reason", None)
    usage = getattr(response, "usage", None)

    completion_avg_logprob, completion_logprob_sum, completion_logprob_count = _extract_completion_logprob_metrics(
        choice0
    )

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


def run_model_text(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0,
):

    return run_model(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )["text"]
