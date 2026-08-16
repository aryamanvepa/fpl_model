"""Two interchangeable LLM backends for Model 4's synthesis step: a local
Ollama model (free, private, needs Ollama installed and running) and a cloud
Anthropic model (needs ANTHROPIC_API_KEY). Both expose the same
`synthesize(prompt) -> str` shape so the rest of the pipeline doesn't care
which one produced the answer. Plain `requests` calls, no SDK dependency.
"""

import os

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


def call_ollama(prompt: str, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST, timeout: int = 180) -> str:
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
    except requests.ConnectionError as e:
        raise RuntimeError(
            f"Couldn't reach Ollama at {host}. Is it installed and running? "
            f"(`ollama serve`, and `ollama pull {model}` if you haven't already)"
        ) from e
    resp.raise_for_status()
    return resp.json()["response"]


def call_anthropic(prompt: str, model: str = ANTHROPIC_MODEL, api_key: str | None = None, timeout: int = 120) -> str:
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your environment to use the cloud backend."
        )
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


BACKENDS = {"ollama": call_ollama, "anthropic": call_anthropic}


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose one of: {list(BACKENDS)}")
    return BACKENDS[name]
