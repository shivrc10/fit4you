# import requests

# BACKEND_URL = "http://127.0.0.1:8000"

# def generate_plan(profile: str, goal: str) -> dict:
#     payload = {
#         "profile": {"raw_text": profile},
#         "goal": goal,
#     }

#     r = requests.post(
#         f"{BACKEND_URL}/generate-plan",
#         json=payload,
#         timeout=600,
#     )

#     if r.status_code != 200:
#         raise RuntimeError(f"Backend error {r.status_code}: {r.text[:500]}")

#     return r.json()
import requests

BACKEND_URL = "http://127.0.0.1:8000"

def generate_plan(profile: str, goal: str) -> dict:
    payload = {
        "profile": {"raw_text": profile or "{}"},
        "goal": goal or "",
    }

    r = requests.post(
        f"{BACKEND_URL}/generate-plan",
        json=payload,
        timeout=300,  # Reduced timeout
    )

    if r.status_code != 200:
        raise RuntimeError(f"Backend error {r.status_code}: {r.text[:800]}")

    return r.json()
