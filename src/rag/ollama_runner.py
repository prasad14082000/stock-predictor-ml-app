# src/rag/ollama_runner.py
import requests
from typing import Optional, Dict, Any

OLLAMA_API = "http://localhost:11434/api/generate"

def generate(
    prompt: str,
    model: str = "llama3",
    options: Optional[Dict[str, Any]] = None,
    stream: bool = False,
) -> str:
    """
    Call a local Ollama model. Returns the final text (non-streaming).
    Raises on HTTP errors with a useful message.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }
    if options:
        payload["options"] = options

    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        if "response" not in data:
            raise RuntimeError(f"Ollama returned unexpected payload: {data}")
        return data["response"]
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Make sure `ollama serve` is running."
        ) from e
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error {r.status_code}: {r.text}") from e
