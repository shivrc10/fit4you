import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def ollama_chat(model: str, system: str, user: str, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": 900,
        },
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=(10, 600))

    # If Ollama returned an error, show it
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")

    # Parse JSON
    data = r.json()

    # Debug: show what keys exist if format differs
    if "message" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {str(data)[:800]}")

    content = data["message"].get("content", "")
    return content or ""
