# from flask import Flask, request, jsonify
# import json

# from ollama_client import ollama_chat

# app = Flask(__name__)

# MODELS = {
#     "planner": "llama3.2:latest",
#     "alternative": "llama3.2:latest",
#     "safety": "llama3.2:latest",
#     "summarizer": "llama3.2:latest",
# }

# # -------------------------------
# # Utilities
# # -------------------------------

# def extract_json(text: str):
#     text = (text or "").strip()
#     if text.startswith("{") and text.endswith("}"):
#         return json.loads(text)

#     start = text.find("{")
#     end = text.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         return json.loads(text[start:end + 1])

#     raise ValueError("No JSON object found")

# def llm_json_with_retry(model, system, prompt, temperature=0.0, retries=2):
#     last_error = None

#     for _ in range(retries):
#         raw = ollama_chat(model, system, prompt, temperature)
#         try:
#             return extract_json(raw)
#         except Exception as e:
#             last_error = e
#             prompt = f"""
# Fix the following into valid JSON only.
# Return ONE JSON object with keys:
# plan_primary, plan_alternatives, safety, questions_to_user, metadata.
# plan_alternatives must be a list.

# INVALID:
# {raw}

# Return corrected JSON only.
# """

#     raise RuntimeError(f"JSON repair failed: {last_error}")

# # -------------------------------
# # Routes
# # -------------------------------

# # @app.post("/generate-plan")
# # def generate_plan():
# #     data = request.get_json(force=True)

# #     profile = data.get("profile", {})
# #     goal = data.get("goal", "")

# #     system_json = "Return ONLY valid JSON. Start with { and end with }."

# #     planner_prompt = f"""
# # User profile:
# # {json.dumps(profile, ensure_ascii=False)}

# # Goal:
# # {goal}

# # Create a 2-week knee-friendly fitness plan.
# # """

# #     planner_raw = ollama_chat(
# #         MODELS["planner"], system_json, planner_prompt, 0.2
# #     )

# #     summarizer_prompt = f"""
# # Merge into final app schema.

# # Planner JSON:
# # {planner_raw}

# # Return ONLY JSON with keys:
# # plan_primary, plan_alternatives, safety, questions_to_user, metadata
# # """

# #     final = llm_json_with_retry(
# #         MODELS["summarizer"], system_json, summarizer_prompt, 0.0
# #     )

# #     final.setdefault("metadata", {})
# #     final["metadata"]["models_used"] = MODELS

# #     return jsonify(final)

# @app.post("/generate-plan")
# def generate_plan():
#     try:
#         data = request.get_json(force=True)
#         profile = data.get("profile", {})
#         goal = data.get("goal", "")

#         system_json = "Return ONLY valid JSON. No markdown. Start with { and end with }."

#         planner_user = f"""
# User profile JSON:
# {json.dumps(profile, ensure_ascii=False)}

# Goal:
# {goal}

# Create a 2-week plan + rules to extend to weeks 3–8 (do NOT write weeks 3–8).
# Must be knee-friendly if injuries include knee pain.
# Return ONLY JSON with keys:
# plan_name, week_1, week_2, progression_rules_weeks_3_to_8, warmup, cooldown, nutrition_timing, habits, questions
# """

#         alt_user = f"""
# User profile JSON:
# {json.dumps(profile, ensure_ascii=False)}

# Goal:
# {goal}

# Create TWO alternatives (2-week plans) that match EXACTLY the same schema as the planner output:
# plan_name, week_1, week_2, progression_rules_weeks_3_to_8, warmup, cooldown, nutrition_timing, habits, questions

# Constraints:
# - Plan B: time-efficient (short sessions)
# - Plan C: equipment-light (bodyweight/bands)

# Rules:
# - No numeric weights (use RPE 1-10 or light/medium/heavy).
# - Keep it knee-friendly if injuries include knee pain.

# Return ONLY JSON with key:
# alternatives (list of exactly 2 plans)
# """

#         safety_user = f"""
# User profile JSON:
# {json.dumps(profile, ensure_ascii=False)}

# Goal:
# {goal}

# Safety check: risks, safer substitutions, stop signs, recovery notes.
# Return ONLY JSON with keys:
# risks, safer_substitutions, stop_signs, recovery_notes
# """

#         # ----------------------------
#         # Run agents (sequential)
#         # ----------------------------
#         plan_a_raw = ollama_chat(MODELS["planner"], system_json, planner_user, 0.2)
#         alts_raw   = ollama_chat(MODELS["alternative"], system_json, alt_user, 0.3)
#         safety_raw = ollama_chat(MODELS["safety"], system_json, safety_user, 0.2)

#         # ----------------------------
#         # Summarizer
#         # ----------------------------
#         summarizer_user = f"""
# Merge these JSON blobs into ONE final app response.

# Planner JSON:
# {plan_a_raw}

# Alternatives JSON:
# {alts_raw}

# Safety JSON:
# {safety_raw}

# Return ONLY JSON with keys:
# plan_primary, plan_alternatives, safety, questions_to_user, metadata
# - questions_to_user: max 4 unique questions total
# - metadata: include models_used
# """

#         final = llm_json_with_retry(
#             MODELS["summarizer"],
#             system_json,
#             summarizer_user,
#             0.0
#         )

#         # ----------------------------
#         # Final safety normalization
#         # ----------------------------
#         if not isinstance(final, dict):
#             raise ValueError("Final output is not a JSON object")

#         final.setdefault("plan_primary", {})
#         final.setdefault("plan_alternatives", [])
#         final.setdefault("safety", {})
#         final.setdefault("questions_to_user", [])
#         final.setdefault("metadata", {})
#         final["metadata"]["models_used"] = MODELS

#         return jsonify(final)

#     except Exception as e:
#         return jsonify({
#             "error": "Final output not valid JSON",
#             "exception": str(e),
#             "debug": {
#                 "plan_a_preview": (plan_a_raw or "")[:700] if "plan_a_raw" in locals() else "",
#                 "alts_preview": (alts_raw or "")[:700] if "alts_raw" in locals() else "",
#                 "safety_preview": (safety_raw or "")[:700] if "safety_raw" in locals() else "",
#             }
#         }), 500


# @app.get("/health")
# def health():
#     return {"status": "ok"}



# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=8000, debug=True)
from flask import Flask, request, jsonify
import json
from concurrent.futures import ThreadPoolExecutor

from ollama_client import ollama_chat  # Fixed import

app = Flask(__name__)
pool = ThreadPoolExecutor(max_workers=1)  # Stable on Mac/Windows

MODELS = {
    "planner": "llama3.2:latest",
    "alternative": "gemma:2b",
    "safety": "gemma:2b", 
    "summarizer": "llama3.2:latest",
}

def extract_json(text: str):
    """Robust JSON extraction"""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")
    
    # Clean common LLM artifacts
    text = text.replace('```json', '').replace('```', '').strip()
    
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end+1])
    
    raise ValueError(f"No JSON object found: {text[:200]}")

def llm_json_with_retry(model, system, prompt, temperature=0.0, retries=3):
    """Retry JSON generation with auto-repair"""
    last_raw = None
    
    for attempt in range(retries):
        raw = ollama_chat(model, system, prompt, temperature)
        last_raw = raw
        
        try:
            return extract_json(raw)
        except Exception:
            if attempt == retries - 1:
                raise ValueError(f"JSON repair failed after {retries} attempts")
            
            # Auto-repair prompt
            repair_prompt = f"""
INVALID JSON from previous attempt:
{raw[:1000]}

FIX into ONE COMPLETE JSON with EXACTLY these top-level keys:
{{"plan_primary": {{}}, "plan_alternatives": [], "safety": {{}}, "questions_to_user": [], "metadata": {{}}}}

- plan_alternatives MUST be a LIST
- Return ONLY valid JSON. No markdown. No explanations.
"""
            prompt = repair_prompt
    
    raise ValueError("Final repair attempt failed")

@app.post("/generate-plan")
def generate_plan_endpoint():
    try:
        data = request.get_json(force=True)
        profile = data.get("profile", {})
        goal = data.get("goal", "")

        system_json = "Return ONLY valid JSON. No markdown. No extra text. Start with { and end with }."

        print(f"🤖 New request: {goal[:50]}...")

        # Planner: Creates primary 2-week plan
        planner_user = f"""
User profile JSON:
{json.dumps(profile, ensure_ascii=False)}

Goal:
{goal}

Create a 2-week knee-friendly fitness plan + progression rules.
Return ONLY JSON with these EXACT keys:
plan_name, week_1, week_2, progression_rules_weeks_3_to_8, warmup, cooldown, nutrition_timing, habits, questions
"""

        # Alternatives: 2 variants (time-efficient, equipment-light)
        alt_user = f"""
User profile JSON:
{json.dumps(profile, ensure_ascii=False)}

Goal:
{goal}

Create TWO alternatives matching EXACT planner schema:
- Plan B: time-efficient (15-20min sessions)
- Plan C: equipment-light (bodyweight/bands only)

Return ONLY JSON: {{"alternatives": [planB, planC]}} (exactly 2 plans)
"""

        # Safety: Risk analysis + substitutions
        safety_user = f"""
User profile JSON:
{json.dumps(profile, ensure_ascii=False)}

Goal:
{goal}

Safety analysis. Return ONLY JSON:
{{
  "risks": ["list risks"],
  "safer_substitutions": {{"exercise": "safer version"}},
  "stop_signs": ["immediate stop conditions"],
  "recovery_notes": ["recovery guidance"]
}}
"""

        # Sequential execution (stable)
        print("🤖 1/4 Planner...")
        plan_a_raw = ollama_chat(MODELS["planner"], system_json, planner_user, 0.2)
        
        print("🤖 2/4 Alternatives...")
        alts_raw = ollama_chat(MODELS["alternative"], system_json, alt_user, 0.3)
        
        print("🤖 3/4 Safety...")
        safety_raw = ollama_chat(MODELS["safety"], system_json, safety_user, 0.2)

        # Summarizer: Perfect final schema
        summarizer_user = f"""
Merge these into ONE final app response:

Planner: {plan_a_raw}
Alternatives: {alts_raw}
Safety: {safety_raw}

Return ONLY JSON with EXACT keys:
{{
  "plan_primary": planner_plan,
  "plan_alternatives": alternatives_list,
  "safety": safety_object, 
  "questions_to_user": ["max 4 unique questions"],
  "metadata": {{}}
}}
"""

        print("🤖 4/4 Summarizer...")
        final = llm_json_with_retry(
            MODELS["summarizer"], 
            system_json, 
            summarizer_user, 
            0.0
        )

        # Final normalization
        final.setdefault("plan_primary", {})
        final.setdefault("plan_alternatives", [])
        final.setdefault("safety", {})
        final.setdefault("questions_to_user", [])
        final.setdefault("metadata", {})
        final["metadata"]["models_used"] = MODELS

        print("✅ SUCCESS - Plan delivered!")
        return jsonify(final)

    except Exception as e:
        print(f"❌ ERROR: {str(e)[:100]}")
        return jsonify({
            "error": "Generation failed",
            "exception": str(e),
            "debug": {
                "profile": str(profile)[:200],
                "goal": goal[:100]
            }
        }), 500

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": "2026-01-17"}

@app.post("/revise-plan")
def revise_plan():
    """Revise existing plan based on user feedback"""
    try:
        data = request.get_json(force=True)
        previous_output = data.get("previous_output", {})
        feedback = data.get("feedback", "").strip()

        if not previous_output:
            return jsonify({"error": "previous_output required"}), 400
        if not feedback:
            return jsonify({"error": "feedback required"}), 400

        system_json = "Return ONLY valid JSON. No markdown."

        revise_prompt = f"""
Previous plan JSON (defines REQUIRED schema):
{json.dumps(previous_output, ensure_ascii=False)}

Apply this feedback:
{feedback}

RULES:
1. Keep EXACT top-level keys: plan_primary, plan_alternatives, safety, questions_to_user, metadata
2. plan_alternatives MUST be LIST (never object)
3. Preserve week_1/week_2 structure
4. Knee-friendly: no jumping if knee pain present
5. No numeric weights (RPE 1-10 or light/medium/heavy)

Return ONLY revised JSON matching original schema.
"""

        revised = llm_json_with_retry(
            MODELS["summarizer"], 
            system_json, 
            revise_prompt, 
            0.0
        )

        revised.setdefault("metadata", {})
        revised["metadata"]["models_used"] = MODELS
        revised["metadata"]["revision_applied"] = feedback[:100]
        
        return jsonify(revised)

    except Exception as e:
        return jsonify({
            "error": "Revision failed",
            "exception": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
