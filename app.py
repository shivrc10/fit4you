from flask import Flask, request, jsonify
import json
from concurrent.futures import ThreadPoolExecutor

from agents import ollama_chat

app = Flask(__name__)
pool = ThreadPoolExecutor(max_workers=1)  # keep 1 for Mac stability

MODELS = {
    "planner": "llama3.2:latest",
    "alternative": "gemma:2b",
    "safety": "gemma:2b",
    "summarizer": "llama3.2:latest",
}

def extract_json(text: str):
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end+1])
    raise ValueError("No JSON object found")

def llm_json_with_retry(model, system, prompt, temperature=0.0):
    out1 = ollama_chat(model, system, prompt, temperature)
    try:
        return extract_json(out1)
    except Exception:
        repair = f"""
Fix the following into valid JSON only.
Also ensure it contains ONLY these top-level keys:
plan_primary, plan_alternatives, safety, questions_to_user, metadata.
plan_alternatives must be a list.

INVALID:
{out1}

Return corrected JSON only.
"""

@app.post("/generate-plan")
def generate_plan():
    data = request.get_json(force=True)
    profile = data.get("profile", {})
    goal = data.get("goal", "")

    system_json = "Return ONLY valid JSON. No markdown. Start with { and end with }."

    planner_user = f"""
User profile JSON:
{json.dumps(profile, ensure_ascii=False)}

Goal:
{goal}

Create a 2-week plan + rules to extend to weeks 3–8 (do NOT write weeks 3–8).
Must be knee-friendly if injuries include knee pain.
Return ONLY JSON with keys:
plan_name, week_1, week_2, progression_rules_weeks_3_to_8, warmup, cooldown, nutrition_timing, habits, questions
"""

    alt_user = f"""
User profile JSON:
{json.dumps(profile, ensure_ascii=False)}

Goal:
{goal}

Create TWO alternatives (2-week plans) that match EXACTLY the same schema as the planner output:
plan_name, week_1, week_2, progression_rules_weeks_3_to_8, warmup, cooldown, nutrition_timing, habits, questions

Constraints:
- Plan B: time-efficient (short sessions)
- Plan C: equipment-light (bodyweight/bands)

Rules:
- No numeric weights (use RPE 1-10 or light/medium/heavy).
- Keep it knee-friendly if injuries include knee pain.

Return ONLY JSON with key:
alternatives (list of exactly 2 plans)
"""


    safety_user = f"""
User profile JSON:
{json.dumps(profile, ensure_ascii=False)}

Goal:
{goal}

Safety check: risks, safer substitutions, stop signs, recovery notes.
Return ONLY JSON with keys:
risks, safer_substitutions, stop_signs, recovery_notes
"""

    # sequential for stability (later we can parallelize)
    plan_a_raw = ollama_chat(MODELS["planner"], system_json, planner_user, 0.2)
    alts_raw   = ollama_chat(MODELS["alternative"], system_json, alt_user, 0.3)
    safety_raw = ollama_chat(MODELS["safety"], system_json, safety_user, 0.2)

    # summarizer merges everything into one clean schema
    summarizer_user = f"""
Merge these JSON blobs into ONE final app response.

Planner JSON:
{plan_a_raw}

Alternatives JSON:
{alts_raw}

Safety JSON:
{safety_raw}

Return ONLY JSON with keys:
plan_primary, plan_alternatives, safety, questions_to_user, metadata
- questions_to_user: max 4 unique questions total
- metadata: include models_used
"""

    # final_raw = ollama_chat(MODELS["summarizer"], system_json, summarizer_user, 0.0)
    final = llm_json_with_retry(MODELS["summarizer"], system_json, summarizer_user, 0.0)

    try:
        final = extract_json(final_raw)
        final.setdefault("metadata", {})
        final["metadata"]["models_used"] = MODELS
        return jsonify(final)
    except Exception as e:
        return jsonify({
            "error": "Final output not valid JSON",
            "exception": str(e),
            "raw_preview": (final_raw or "")[:1500],
            "debug": {
                "plan_a_preview": (plan_a_raw or "")[:700],
                "alts_preview": (alts_raw or "")[:700],
                "safety_preview": (safety_raw or "")[:700],
            }
        }), 500

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/revise-plan")
def revise_plan():
    data = request.get_json(force=True)

    previous_output = data.get("previous_output", {})
    feedback = (data.get("feedback", "") or "").strip()

    if not previous_output:
        return jsonify({"error": "previous_output is required"}), 400
    if not feedback:
        return jsonify({"error": "feedback is required"}), 400

    system_json = "Return ONLY valid JSON. No markdown. Start with { and end with }."

    revise_prompt = f"""
You are an expert editor of fitness plans.

Previous output JSON (this defines the required schema):
{json.dumps(previous_output, ensure_ascii=False)}

Expert feedback to apply:
{feedback}

STRICT RULES (must follow):
1) Return ONLY ONE JSON object.
2) Keep EXACTLY these top-level keys (no new top-level keys):
plan_primary, plan_alternatives, safety, questions_to_user, metadata
3) plan_alternatives MUST be a LIST (array). Never an object.
4) Do NOT change data types of any existing fields.
5) Respect profile constraints implied in the plan:
- days_per_week MUST stay 4
- knee-friendly: no jumping/running if knee pain present
6) Do NOT invent numbers like weights (use RPE 1-10 or "light/medium/heavy").
7) If you need knee flare-up options, add them INSIDE:
plan_primary.week_1 and plan_primary.week_2 (as "flare_up_option")
and similarly for each alternative week_1/week_2.
8) Keep week structure: week_1 and week_2 must remain present in plan_primary.
9) Output must start with {{ and end with }}.

Now revise the plan applying the feedback while preserving the schema.
Return ONLY the revised JSON.
"""


    try:
        revised = llm_json_with_retry(MODELS["summarizer"], system_json, revise_prompt, 0.0)
        revised.setdefault("metadata", {})
        revised["metadata"]["models_used"] = MODELS
        revised["metadata"]["revision_applied"] = True
        return jsonify(revised)
    except Exception as e:
        return jsonify({
            "error": "Revised output not valid JSON",
            "exception": str(e)
        }), 500



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
