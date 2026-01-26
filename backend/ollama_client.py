# import requests

# OLLAMA_URL = "http://localhost:11434/api/chat"

# def ollama_chat(model: str, system: str, user: str, temperature: float = 0.1) -> str:
#     payload = {
#         "model": model,
#         "format": "json",
#         "messages": [
#             {"role": "system", "content": system},
#             {"role": "user", "content": user},
#         ],
#         "options": {
#             "temperature": temperature,
#             "num_predict": 300,
#         },
#         "stream": False,
#     }

#     r = requests.post(OLLAMA_URL, json=payload, timeout=(10, 120))
#     if r.status_code != 200:
#         raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")

#     data = r.json()
#     return data.get("message", {}).get("content", "")
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"

def ollama_chat(model: str, system: str, user: str, temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        # REMOVED "format": "json" - let manual extraction handle it
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": 600,  # Increased to avoid truncation
        },
        "stream": False,
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=(10, 180))
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()
    return data.get("message", {}).get("content", "")
