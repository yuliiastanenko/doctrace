import time
from dataclasses import dataclass
from functools import lru_cache

from doctrace.config import settings

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "cloudflare": ("https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"),
    "anthropic": ("https://api.anthropic.com/v1/", "claude-haiku-5-5"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5:7b"),
}


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float


def provider_config(provider: str | None = None, model: str | None = None) -> tuple[str, str, str]:
    provider = provider or settings.llm_provider
    if provider not in PROVIDERS:
        raise SystemExit(f"unknown LLM_PROVIDER {provider!r}, choose one of {', '.join(PROVIDERS)}")
    base_url, default_model = PROVIDERS[provider]
    if "{account}" in base_url:
        if not settings.cloudflare_account_id:
            raise SystemExit("CLOUDFLARE_ACCOUNT_ID is empty, check your .env file")
        base_url = base_url.format(account=settings.cloudflare_account_id)
    api_key = settings.llm_api_key or ("ollama" if provider == "ollama" else "")
    if not api_key:
        raise SystemExit("LLM_API_KEY is empty, check your .env file")
    return base_url, api_key, model or settings.llm_model or default_model


@lru_cache
def _client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=6)


def list_models(provider: str | None = None) -> list[str]:
    base_url, api_key, _ = provider_config(provider)
    return sorted(m.id for m in _client(base_url, api_key).models.list())


def complete(messages: list[dict], provider: str | None = None, model: str | None = None) -> Completion:
    base_url, api_key, model = provider_config(provider, model)
    start = time.perf_counter()
    response = _client(base_url, api_key).chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=4000,
    )
    usage = response.usage
    return Completion(
        text=response.choices[0].message.content or "",
        model=model,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        seconds=time.perf_counter() - start,
    )
