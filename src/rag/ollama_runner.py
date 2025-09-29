# src/rag/ollama_runner.py
import time
import requests
from typing import Optional, Dict, Any

OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_BASE = "http://localhost:11434"

def ping_ollama(base_url: str = OLLAMA_BASE, timeout: float = 3.0) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=timeout)
        r.raise_for_status()
        return True
    except Exception:
        return False

def generate(
    prompt: str,
    model: str = "llama3",
    options: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    read_timeout: float = 600.0,   # longer for first load
    connect_timeout: float = 5.0,
    max_retries: int = 2,
) -> str:
    payload = {"model": model, "prompt": prompt, "stream": stream}
    if options: payload["options"] = options

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(
                OLLAMA_API,
                json=payload,
                timeout=(connect_timeout, read_timeout),  # (connect, read)
            )
            r.raise_for_status()
            data = r.json()
            if "response" not in data:
                raise RuntimeError(f"Ollama returned unexpected payload: {data}")
            return data["response"]
        except requests.exceptions.ReadTimeout as e:
            last_err = e
            # brief backoff; first run can be slow while the model loads
            time.sleep(2 * (attempt + 1))
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                "Could not reach Ollama at http://localhost:11434. "
                "Make sure `ollama serve` is running."
            ) from e
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error {r.status_code}: {r.text}") from e

    raise RuntimeError(
        f"Ollama timed out after {max_retries+1} attempt(s). "
        "Pre-warm by running `ollama run llama3 \"hi\"` in a terminal."
    ) from last_err
