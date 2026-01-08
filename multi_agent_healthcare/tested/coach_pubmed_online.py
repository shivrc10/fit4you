"""
Coach agent with online PubMed retrieval.
- Uses NCBI Entrez API to fetch abstracts in real-time.
- Integrates with local HF causal LLM (LLaMA/TinyLlama) to generate evidence-based answers/plans.
- Implements simple ITR loop (Critic -> Coach -> Validator).
- Avoids hallucinations by grounding in retrieved abstracts.
"""

import os
import re
import torch
from typing import List, Dict
from Bio import Entrez
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# CONFIG
# -------------------------
Entrez.email = "your_email@example.com"  # REQUIRED by NCBI

# -------------------------
# Retriever: fetch from PubMed online
# -------------------------
def fetch_pubmed(query: str, max_results=5) -> List[Dict]:
    """
    Fetch PubMed abstracts for a query.
    Returns a list of dicts: {"pmid": str, "title": str, "text": str}
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]

    abstracts = []
    for pmid in ids:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        text = handle.read()
        handle.close()
        # Split into title + abstract
        lines = text.strip().split("\n\n")
        title = lines[0] if len(lines) > 0 else ""
        abstract_text = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]
        abstracts.append({"pmid": pmid, "title": title, "text": abstract_text})
    return abstracts

def build_context(retrieved: List[Dict], max_chars=1500) -> str:
    parts = []
    total = 0
    for r in retrieved:
        txt = r.get("text","")
        title = r.get("title","")
        snippet = f"{title} — {txt}" if title else txt
        if total + len(snippet) > max_chars:
            break
        parts.append(f"- {snippet[:800].replace('\n',' ')}")
        total += len(snippet)
    if not parts:
        return "No relevant scientific abstracts found."
    return "\n".join(parts)

# -------------------------
# Local LLM wrapper
# -------------------------
class LocalGenerator:
    def __init__(self, model_path: str, device='cpu', max_length=512):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device!='cpu' else torch.float32,
            device_map="auto" if device!='cpu' else None
        )
        self.max_length = max_length

    def generate(self, prompt, max_new_tokens=256, temperature=0.2, top_p=0.95):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            top_p=top_p
        )
        out = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return out[len(prompt):].strip() if out.startswith(prompt) else out.strip()

# -------------------------
# Prompt builder
# -------------------------
SYSTEM_PROMPT = """You are CoachGPT, an evidence-based fitness and nutrition coach.
- Use only the evidence given in CONTEXT.
- Provide safe, practical, non-judgmental advice and a simple plan.
- Include safety notes and explicitly state uncertainty if evidence is weak.
"""

def build_prompt(user_query: str, context_text: str) -> str:
    return SYSTEM_PROMPT + f"\nCONTEXT:\n{context_text}\n\nUSER QUERY:\n{user_query}\n\nTASK:\nProvide an evidence-based short answer and plan. Cite retrieved abstracts. Answer:\n"

# -------------------------
# Critic
# -------------------------
def critic_check(draft_text: str, user_profile: Dict) -> List[str]:
    issues = []
    if user_profile.get("fitness_level","").lower()=="beginner":
        if re.search(r'60\s*min|90\s*min|high intensity|advanced|heavy load', draft_text, re.I):
            issues.append("Plan/intensity may be too high for a beginner.")
    if 'knee' in ''.join(user_profile.get("injuries",[])).lower():
        if re.search(r'jump|plyometric|sprint', draft_text, re.I):
            issues.append("High-impact moves unsafe for knee injury.")
    return issues

# -------------------------
# Validator
# -------------------------
def validator_check(draft_text: str, user_profile: Dict) -> (bool, List[str]):
    reasons = []
    ok = True
    age = user_profile.get("age")
    if age and age >= 75:
        reasons.append("Age >=75: recommend medical clearance.")
        ok = False
    if user_profile.get("pregnant"):
        reasons.append("Pregnancy: consult OB/GYN and adapt plan.")
        ok = False
    return ok, reasons

# -------------------------
# ITR loop
# -------------------------
def run_itr(user_query: str, user_profile: Dict, generator: LocalGenerator, top_k=5, max_rounds=3):
    retrieved = fetch_pubmed(user_query, max_results=top_k)
    context_text = build_context(retrieved)
    prompt = build_prompt(user_query, context_text)

    trace = {'user_query': user_query, 'retrieved_docs': retrieved, 'rounds': []}
    draft = generator.generate(prompt, max_new_tokens=400)
    for r in range(1, max_rounds+1):
        issues = critic_check(draft, user_profile)
        trace['rounds'].append({'round': r, 'draft': draft, 'critic_issues': issues})
        if not issues:
            break
        revision_prompt = SYSTEM_PROMPT + f"\nCONTEXT:\n{context_text}\nCRITIC ISSUES:\n" + "\n".join(issues) + f"\nPREVIOUS DRAFT:\n{draft}\n\nTASK:\nRevise draft addressing critic issues. Be concise and evidence-based.\n\nRevised Answer:\n"
        draft = generator.generate(revision_prompt, max_new_tokens=400)
    ok, reasons = validator_check(draft, user_profile)
    trace['final'] = {'draft': draft, 'validator_ok': ok, 'validator_reasons': reasons}
    return trace

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    generator = LocalGenerator(args.model_path, device=args.device)
    print("Coach agent ready. Type 'exit' to quit.")
    while True:
        q = input("\nUser query> ").strip()
        if not q:
            continue
        if q.lower() in ("exit","quit"):
            break
        user_profile = {}
        fl = input("Fitness level (beginner/intermediate/advanced) [beginner]> ").strip() or "beginner"
        user_profile['fitness_level'] = fl
        age = input("Age [empty]> ").strip()
        user_profile['age'] = int(age) if age.isdigit() else None
        inj = input("Any injuries (comma separated) [none]> ").strip()
        user_profile['injuries'] = [x.strip() for x in inj.split(',')] if inj else []
        preg = input("Pregnant? (y/n) [n]> ").strip().lower()
        user_profile['pregnant'] = (preg == 'y')

        trace = run_itr(q, user_profile, generator, top_k=args.top_k)
        print("\n--- Retrieved abstracts ---")
        for d in trace['retrieved_docs']:
            print(f"{d['title'][:50]}... — {d['text'][:200]}...")
        print("\n--- Final Draft ---")
        print(trace['final']['draft'])
        print("\n--- Validator ---")
        print("OK:", trace['final']['validator_ok'])
        print("Reasons:", trace['final']['validator_reasons'])
        print("\n--- End ---")
