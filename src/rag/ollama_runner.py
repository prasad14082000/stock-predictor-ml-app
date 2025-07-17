# src/rag/ollama_runner.py

import requests

def query_ollama(prompt, model="llama3", base_url="http://localhost:11434"):
    """
    Sends a prompt to the Ollama LLM server and returns the response text.

    Args:
        prompt (str): The prompt/question to send to Ollama.
        model (str): Name of the Ollama model (default: "llama3").
        base_url (str): Base URL for Ollama server (default: localhost).

    Returns:
        str: The LLM-generated response (answer text).
    """
    url = f"{base_url}/api/generate"
    try:
        response = requests.post(url, json={"model": model, "prompt": prompt})
        response.raise_for_status()
        data = response.json()
        # 'response' key holds the generated text in Ollama's API
        return data.get("response", "").strip()
    except Exception as e:
        return f"Ollama API error: {str(e)}"
