#new
# fitness_advisor_final_272k_pubmed_interactive_summary.py
# 272k PubMedQA + Real Citations + Groq LLMs + Graph-of-Thoughts + Post-Evidence Summary + User Agree/Regenerate

import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import List, Dict, TypedDict
from rich.console import Console
from rich.panel import Panel
import chromadb
from tqdm import tqdm
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm
from textwrap import wrap

import re

console = Console()

DEV_MODE = True

# ================================
# 1. Build 272k PubMed Vector DB (runs once)
# ================================
DB_PATH = "./pubmed_272k_db"
os.makedirs(DB_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=DB_PATH)
COLLECTION_NAME = "pubmed_272k"
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
    console.print("[bold yellow]Building FULL 272k PubMedQA vector database (first run only, ~10–15 mins)...[/bold yellow]")
    collection = client.create_collection(COLLECTION_NAME)
   
    files = ["pubmed272k/ori_pqac.json", "pubmed272k/ori_pqal.json", "pubmed272k/ori_pqau.json"]
    all_data = {}
   
    for f in files:
        if os.path.exists(f):
            with open(f) as jf:
                data = json.load(jf)
                all_data.update(data)
                console.print(f"[green]Loaded[/green] {f} → {len(data):,} entries")
   
    console.print(f"[bold blue]Indexing {len(all_data):,} real PubMed papers...[/bold blue]")
   
    batch_size = 64
    batch_docs = []
    batch_ids = []
    batch_texts = []
   
    for pid, item in tqdm(all_data.items(), desc="Embedding 272k papers"):
        text = ""
        if "QUESTION" in item:
            text += f"Question: {item['QUESTION']}\n"
        if "CONTEXTS" in item:
            text += "Context: " + " ".join(item["CONTEXTS"]) + "\n"
        if "LONG_ANSWER" in item:
            text += f"Answer: {item['LONG_ANSWER']}\n"
        if "final_decision" in item:
            text += f"Conclusion: {item['final_decision']}\n"
       
        batch_texts.append(text)
        batch_ids.append(pid)
        batch_docs.append({
            "question": item.get("QUESTION", "")[:200],
            "answer": item.get("LONG_ANSWER", "")[:300]
        })
       
        if len(batch_texts) >= batch_size:
            embeddings = embedder.embed_documents(batch_texts)
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_docs
            )
            batch_texts, batch_ids, batch_docs = [], [], []
   
    if batch_texts:
        embeddings = embedder.embed_documents(batch_texts)
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_docs
        )
   
    console.print("[bold green]FULL 272k PubMedQA database ready! Real citations enabled![/bold green]")
else:
    collection = client.get_collection(COLLECTION_NAME)
    console.print("[bold green]Loaded existing 272k PubMed database[/bold green]")

# ================================
# 2. LLMs (Groq API)
# ================================
doctor_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
critic_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.7)
supporter_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.9)
coach_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
summarizer_llm = doctor_llm  # Reuse for quick, evidence-based summaries

# ================================
# 3. Agent State
# ================================
class AgentState(TypedDict):
    user_question: str
    user_profile: str
    thoughts: List[Dict]
    final_answer: str
    evidence_complete: bool  # Flag for flow control

# ================================
# 4. Doctor Node
# ================================


def doctor_node(state: AgentState):
    results = collection.query(
        query_texts=[state['user_question'] + " fitness exercise health impact"],
        n_results=6
    )
   
    context_blocks = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        pmid = results['ids'][0][i]
        context_blocks.append(f"[PMID: {pmid}] {doc.strip()[:1200]}...")
   
    context = "\n\n".join(context_blocks)
   
    prompt = f"""You are Evidence Retrieval Agent using the FULL 272k PubMedQA dataset.
Use ONLY the real studies below. Cite them as [PMID: 12345678].
Retrieved evidence:
{context}
User: {state['user_profile']}
Goal: {state['user_question']}
Give only medical facts, safety profile, and evidence level.
Never give training plans.
Output exactly:
EVIDENCE RETRIEVAL: [your response with real PMID citations]"""
   
    resp = doctor_llm.invoke(prompt)
    sources_str = " | ".join([f"[bold cyan]PMID:{pid}[/bold cyan]" for pid in results['ids'][0][:4]])
    console.print(Panel(resp.content + f"\n\nReal sources → {sources_str}",
                        title="[blue]Evidence Retrieval – 272k Real PubMed[/blue]", border_style="blue"))
   
    thought = {
        "id": len(state["thoughts"])+1,
        "agent": "Evidence Retrieval Agent",
        "content": resp.content,
        "sources": results['ids'][0][:4]
    }
   
    return {"thoughts": state["thoughts"] + [thought]}

# ================================
# 5. Critic Node
# ================================
def critic_node(state: AgentState):
    prev = state["thoughts"][-1]["content"]
    prompt = f"""You are Validator Agent. Attack weak evidence, small samples, bias, or overgeneralization.
Evidence Retrieval claimed (with real PMIDs):
{prev}
Tear it apart if needed. Be ruthless but fair.
Validator: """
   
    resp = critic_llm.invoke(prompt)
    console.print(Panel(resp.content, title="[red]Validator[/red]", border_style="red"))
   
    thought = {"id": len(state["thoughts"])+1, "agent": "Critic", "content": resp.content}
    return {"thoughts": state["thoughts"] + [thought]}

# ================================
# 6. Supporter Node
# ================================
def supporter_node(state: AgentState):
    prompt = f"""You are Supporter. Highlight benefits, real success patterns, and hope.
User profile: {state['user_profile']}
Goal: {state['user_question']}
Previous debate: {json.dumps([t["content"][:500] for t in state["thoughts"]], indent=2)}
Be encouraging and realistic.
SUPPORTER: """
   
    resp = supporter_llm.invoke(prompt)
    console.print(Panel(resp.content, title="[green]Supporter[/green]", border_style="green"))
   
    thought = {"id": len(state["thoughts"])+1, "agent": "Supporter", "content": resp.content}
    return {"thoughts": state["thoughts"] + [thought]}

# ================================
# 7. Summary Node (new)
# ================================
def summary_node(state: AgentState):
    debate = "\n\n".join([f"{t['agent']}: {t['content'][:400]}" for t in state["thoughts"]])
    prompt = f"""Summarize the full evidence debate in EXACTLY 150 words or less.
Use bullet points (one per agent).
Be neutral, factual, and concise.
Highlight key pros/cons, evidence strength, and implications for the user's goal.
Debate:
{debate}
Output ONLY the summary bullets, no intro/outro.
"""
   
    resp = summarizer_llm.invoke(prompt)
    summary_text = resp.content.strip()
   
    console.print(Panel(summary_text, title="[bold magenta]Evidence Debate Summary (150 words max)[/bold magenta]", border_style="magenta"))
   
    while True:
        choice = input("\nDo you agree with this summary and want to proceed to your personalized plan? (y/n): ").strip().lower()
        if choice in ["y", "n"]:
            break
   
    if choice == "y":
        return {
            "evidence_complete": True,
            "summary": summary_text
        }
    else:
        console.print("[yellow]Regenerating evidence debate...[/yellow]\n")
        return {"thoughts": [], "evidence_complete": False}  # Reset for regeneration
    
    
    

def hallucination_score(
    agent: str,
    text: str,
    doctor_text: str,
    summary_text: str,
    evidence_vocab: set
) -> float:
    """
    Independent hallucination score per agent
    """

    tokens = tokenize(text)
    claim_tokens = [t for t in tokens if t in CLAIM_KEYWORDS]

    # No claims → no hallucination
    if not claim_tokens:
        return 0.0

    # --- Doctor ---
    if agent == "Doctor":
        unsupported = [t for t in claim_tokens if t not in evidence_vocab]
        return len(unsupported) / len(claim_tokens)

    # --- Summary ---
    if agent == "Summary":
        doctor_vocab = set(tokenize(doctor_text))
        unsupported = [t for t in claim_tokens if t not in doctor_vocab]
        return len(unsupported) / len(claim_tokens)

    # --- Critic ---
    if agent == "Critic":
        # Critic should not assert facts at all
        return len(claim_tokens) / len(tokens)

    # --- Supporter ---
    if agent == "Supporter":
        # Supporter should not assert facts
        return len(claim_tokens) / len(tokens)

    # --- Coach ---
    if agent == "Coach":
        unsupported = [t for t in claim_tokens if t not in evidence_vocab]
        return len(unsupported) / len(claim_tokens)

    return 0.0
def compute_independent_hallucination_scores(thoughts, summary_text=""):
    scores = {}

    doctor_text = next(
        (t["content"] for t in thoughts if t["agent"] == "Doctor"),
        ""
    )

    evidence_vocab = extract_evidence_vocab(thoughts, summary_text)

    for t in thoughts:
        agent = t["agent"]
        text = t["content"]

        score = hallucination_score(
            agent=agent,
            text=text,
            doctor_text=doctor_text,
            summary_text=summary_text,
            evidence_vocab=evidence_vocab
        )

        scores[agent] = round(score, 3)

    return scores


ROLE_INSTRUCTIONS = {
    "Doctor": {
        "required": [
            "pmid", "study", "evidence", "clinical", "risk"
        ],
        "forbidden": [
            "week", "schedule", "sets", "reps", "routine", "habit"
        ]
    },
    "Critic": {
        "required": [
            "bias", "limitation", "sample", "confound", "overgeneral"
        ],
        "forbidden": [
            "recommend", "plan", "should do", "workout", "diet"
        ]
    },
    "Supporter": {
        "required": [
            "encourag", "positive", "realistic", "support"
        ],
        "forbidden": [
            "study", "pmid", "risk", "clinical", "trial"
        ]
    },
    "Summary": {
        "required": [
            "pros", "cons", "evidence", "strength"
        ],
        "forbidden": [
            "recommend", "plan", "schedule", "should"
        ]
    },
    "Coach": {
        "required": [
            "week", "schedule", "progress", "habit", "plan"
        ],
        "forbidden": [
            "pmid", "study", "clinical", "trial", "meta-analysis"
        ]
    }
}


def prompt_adherence_score(agent: str, text: str) -> float:
    """
    Measures how well the agent followed its role instructions.
    Score ∈ [0, 1]
    """
    if agent not in ROLE_INSTRUCTIONS or not text:
        return 0.0

    rules = ROLE_INSTRUCTIONS[agent]
    text_l = text.lower()

    # Required signals
    required_hits = sum(
        1 for kw in rules["required"] if kw in text_l
    )
    required_score = required_hits / max(len(rules["required"]), 1)

    # Forbidden signals
    forbidden_hits = sum(
        1 for kw in rules["forbidden"] if kw in text_l
    )
    forbidden_penalty = forbidden_hits / max(len(rules["forbidden"]), 1)

    # Final adherence score
    score = required_score * (1 - forbidden_penalty)

    return round(max(0.0, min(score, 1.0)), 3)

def instruction_drift_rate(adherence_score: float) -> float:
    """
    Drift = how much the agent deviated from instructions
    """
    return round(1.0 - adherence_score, 3)


def compute_prompt_adherence_scores(thoughts):
    adherence = {}
    drift = {}

    for t in thoughts:
        agent = t["agent"]
        text = t["content"]

        pas = prompt_adherence_score(agent, text)
        idr = instruction_drift_rate(pas)

        adherence[agent] = pas
        drift[agent] = idr

    return adherence, drift



# ================================
# 8. Coach Node
# ================================
def coach_node(state: AgentState):

    # 🔒 HARD SAFETY GATE
    # if not state.get("user_agreed", False):
    #     raise RuntimeError(
    #         "Coach node invoked without explicit user consent"
    #     )

    # debate = json.dumps(
    #     [{"agent": t["agent"], "text": t["content"][:1000]} for t in state["thoughts"]],
    #     indent=2
    # )
    
    debate = json.dumps(
        [{"agent": t["agent"], "text": t["content"][:1000]} for t in state["thoughts"]],
        indent=2
    )
    
    
    prompt = f"""You are Action Planning Agent — the practical expert.
Full debate with REAL PubMed evidence:
{debate}
NEVER repeat medical facts, risks, or motivation already said.
Your ONLY job: give a concrete, personalized 4–8 week plan.
User: {state['user_profile']}
Include:
• Weekly schedule table
• Progression rules
• Modifications for injuries
• Nutrition timing
• Habit system
• 2 questions at the end
🧡 ACTION PLAN
"""

    resp = coach_llm.invoke(prompt)
    plan_text = resp.content

    state.setdefault("eval", {})

    # =========================
    # Hallucination Evaluation
    # =========================
    evidence_vocab = extract_evidence_vocab(state["thoughts"])
    
    doctor_text = next(
    (t["content"] for t in state["thoughts"] if t["agent"] == "Doctor"),
    ""
    )

    hall_rate = hallucination_score(
        agent="Coach",
        text=plan_text,
        doctor_text=doctor_text,
        summary_text=state.get("summary", ""),
        evidence_vocab=evidence_vocab
    )

    state["eval"]["hallucination_rate"] = round(hall_rate, 3)

    # =========================
    # LLM-as-Judge
    # =========================
    judge_scores = llm_as_judge(
        profile=state["user_profile"],
        goal=state["user_question"],
        output=plan_text
    )

    state["eval"]["LLM_Judge"] = judge_scores

    # =========================
    # Quality gates
    # =========================
    if (not DEV_MODE) and (
        judge_scores.get("safety", 1) < 4
        or judge_scores.get("overall", 1) < 3
        or hall_rate > 0.30
    ):
        console.print("[red]❌ Plan rejected by quality gate[/red]")
        return {
            "thoughts": state["thoughts"],
            "final_answer": "",
            "eval": state["eval"]
        }

    # =========================
    # Accept path
    # =========================
    console.print(
        Panel(
            f"Hallucination Rate: {hall_rate:.2%}",
            title="[bold red]Hallucination Evaluation[/bold red]",
            border_style="red"
        )
    )

    console.print(
        Panel(
            plan_text,
            title="[bold yellow]Your Final Plan[/bold yellow]",
            border_style="yellow"
        )
    )

    console.print(
        Panel(
            json.dumps(judge_scores, indent=2),
            title="[bold cyan]LLM-as-Judge Scores[/bold cyan]",
            border_style="cyan"
        )
    )

    thought = {
        "id": len(state["thoughts"]) + 1,
        "agent": "Coach",
        "content": plan_text
    }

    return {
        "thoughts": state["thoughts"] + [thought],
        "final_answer": plan_text,
        "eval": state["eval"]
    }


# ================================
# 9. Build Graph-of-Thoughts
# ================================
graph = StateGraph(AgentState)

graph.add_node("doctor", doctor_node)
graph.add_node("critic", critic_node)
graph.add_node("supporter", supporter_node)
graph.add_node("summary", summary_node)
graph.add_node("coach", coach_node)

graph.set_entry_point("doctor")
graph.add_edge("doctor", "critic")
graph.add_edge("critic", "supporter")
graph.add_edge("supporter", "summary")

# Conditional edges from summary
graph.add_conditional_edges(
    "summary",
    lambda state: "proceed" if state.get("evidence_complete", False) else "regenerate",
    {
        "regenerate": "doctor",
        "proceed": "coach"
    }
)

graph.add_edge("coach", END)

app = graph.compile()

# ================================
# PDF Generator
# ================================
def generate_plan_pdf(filename: str, profile: str, goal: str, plan_text: str):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    margin_x = 2.2 * cm
    margin_y = 2.2 * cm
    y = height - margin_y

    def new_page():
        nonlocal y
        c.showPage()
        y = height - margin_y
        c.setFont("Helvetica", 11)

    # =========================
    # Title
    # =========================
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin_x, y, "Personalized Fitness Plan")
    y -= 14

    c.setStrokeColor(colors.grey)
    c.line(margin_x, y, width - margin_x, y)
    y -= 20

    # =========================
    # Profile & Goal
    # =========================
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, "User Profile")
    y -= 14

    c.setFont("Helvetica", 11)
    for line in wrap(profile, 90):
        c.drawString(margin_x, y, line)
        y -= 14

    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, "Goal")
    y -= 14

    c.setFont("Helvetica", 11)
    for line in wrap(goal, 90):
        c.drawString(margin_x, y, line)
        y -= 14

    y -= 18

    # =========================
    # Plan Content
    # =========================
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin_x, y, "Coach’s Action Plan")
    y -= 16

    c.setFont("Helvetica", 11)

    for block in plan_text.split("\n"):
        if y <= margin_y + 40:
            new_page()

        # Headings (heuristic)
        if block.strip().endswith(":") or block.strip().startswith("**"):
            y -= 6
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin_x, y, block.replace("*", ""))
            y -= 14
            c.setFont("Helvetica", 11)
            continue

        # Tables (render as monospaced block)
        if "|" in block:
            c.setFont("Courier", 9)
            c.drawString(margin_x, y, block)
            y -= 12
            c.setFont("Helvetica", 11)
            continue

        # Normal text
        wrapped = wrap(block, 95) if block.strip() else [""]
        for line in wrapped:
            if y <= margin_y + 20:
                new_page()
            c.drawString(margin_x, y, line)
            y -= 14

    # =========================
    # Footer
    # =========================
    y -= 10
    c.setStrokeColor(colors.lightgrey)
    c.line(margin_x, y, width - margin_x, y)
    y -= 14

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        margin_x,
        y,
        "Generated by Fit4You — Evidence-Aware, Human-in-the-Loop Fitness Planning"
    )

    c.save()


# ================================
# 10. LLM as a Judge
# ================================
JUDGE_PROMPT = """
You are a grading function inside a software system.

STRICT RULES:
- Output MUST be valid JSON
- Output MUST contain ONLY JSON
- No explanations
- No markdown
- No extra text
- Numbers only

If you violate these rules, the output will be discarded.

TASK:
Evaluate the FINAL PLAN.

Score each criterion from 1 to 5.
3 = acceptable
5 = excellent
1 = unsafe or unusable

USER PROFILE:
{profile}

USER GOAL:
{goal}

FINAL PLAN:
{output}

OUTPUT FORMAT (EXACT):
{{
  "safety": <int 1-5>,
  "coherence": <int 1-5>,
  "personalization": <int 1-5>,
  "usefulness": <int 1-5>,
  "overall": <float>
}}"""

judge_llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.0
)
def llm_as_judge(profile, goal, output):
    prompt = JUDGE_PROMPT.format(
        profile=profile,
        goal=goal,
        output=output[:6000]
    )

    resp = judge_llm.invoke(prompt)
    # print("\n===== RAW JUDGE OUTPUT =====")
    # print(resp.content)
    # print("===== END RAW JUDGE OUTPUT =====\n")

    try:
        return json.loads(resp.content)
    except Exception:
        return {
            "safety": 1,
            "coherence": 1,
            "personalization": 1,
            "usefulness": 1,
            "overall": 1
        }
        
# ================================
# 11. Evaluation (Role Adherence Eval)
# ================================
ROLE_KEYWORDS = {
    "Doctor": [
        "study", "evidence", "pmid", "clinical", "risk", "meta-analysis"
    ],
    "Critic": [
        "bias", "sample size", "limitation", "confound", "overgeneralization"
    ],
    "Supporter": [
        "benefit", "improvement", "encouraging", "positive", "motivation"
    ],
    "Coach": [
        "week", "schedule", "sets", "reps", "routine", "progression"
    ]
}

def role_adherence_score(agent_output: str, role: str) -> float:
    if not agent_output or role not in ROLE_KEYWORDS:
        return 0.0

    text = agent_output.lower()
    hits = sum(kw in text for kw in ROLE_KEYWORDS[role])
    return hits / len(ROLE_KEYWORDS[role])


# ================================
# Evaluation block
# ================================
STOPWORDS = set([
    "the","and","to","of","in","a","is","for","on","with","as","by",
    "that","this","it","are","be","or","an","at","from","if","can",
    "will","may","should","you","your"
])

def tokenize(text: str):
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]

# ================================
# Claim-aware Hallucination Metric
# ================================

CLAIM_KEYWORDS = {
    "risk", "risks", "increase", "increases", "decrease", "decreases",
    "improve", "improves", "reduce", "reduces",
    "prevent", "prevents", "cause", "causes",
    "associated", "association", "linked", "leads",
    "evidence", "study", "studies", "clinical",
    "trial", "meta-analysis", "significant", "effect", "effects"
}

ROLE_VIOLATIONS = {
    "Doctor": [
        "workout", "training plan", "exercise routine", "sets", "reps",
        "nutrition plan", "meal plan", "diet schedule"
    ],
    "Critic": [
        "study", "studies", "evidence", "pmid", "clinical",
        "risk", "increase", "decrease", "effect"
    ],
    "Supporter": [
        "study", "studies", "evidence", "pmid", "clinical",
        "risk", "increase", "decrease", "effect"
    ],
    "Summary": [
        "new study", "additional evidence", "research shows",
        "clinical trial", "meta-analysis"
    ],
    "Coach": [
        "study", "studies", "evidence", "pmid", "clinical trial",
        "meta-analysis", "research shows"
    ]
}

def violation_rate(agent: str, text: str) -> float:
    """
    Measures explicit forbidden instruction violations per agent.
    Independent of hallucination and prompt adherence.
    """

    if not text or agent not in ROLE_VIOLATIONS:
        return 0.0

    text_l = text.lower()
    forbidden = ROLE_VIOLATIONS[agent]

    # Count violations
    violations = sum(1 for kw in forbidden if kw in text_l)

    tokens = tokenize(text)

    # Normalize to avoid verbosity bias
    return round(violations / max(len(tokens), 1), 3)

def compute_violation_rates(thoughts):
    rates = {}

    for t in thoughts:
        agent = t["agent"]
        text = t["content"]

        rate = violation_rate(agent, text)
        rates[agent] = rate

    return rates



def hallucination_score(
    agent: str,
    text: str,
    doctor_text: str,
    summary_text: str,
    evidence_vocab: set
) -> float:
    """
    Independent, role-aware hallucination score
    """

    tokens = tokenize(text)
    claim_tokens = [t for t in tokens if t in CLAIM_KEYWORDS]

    # No factual claims → no hallucination
    if not claim_tokens:
        return 0.0

    # Doctor must be fully evidence-grounded
    if agent == "Doctor":
        unsupported = [t for t in claim_tokens if t not in evidence_vocab]
        return len(unsupported) / len(claim_tokens)

    # Summary must not introduce new claims
    if agent == "Summary":
        doctor_vocab = set(tokenize(doctor_text))
        unsupported = [t for t in claim_tokens if t not in doctor_vocab]
        return len(unsupported) / len(claim_tokens)

    # Critic should not assert facts at all
    if agent == "Critic":
        return len(claim_tokens) / len(tokens)

    # Supporter should not assert facts
    if agent == "Supporter":
        return len(claim_tokens) / len(tokens)

    # Coach may act, but claims must be grounded
    if agent == "Coach":
        unsupported = [t for t in claim_tokens if t not in evidence_vocab]
        return len(unsupported) / len(claim_tokens)

    return 0.0


def compute_independent_hallucination_scores(thoughts, summary_text=""):
    scores = {}

    doctor_text = next(
        (t["content"] for t in thoughts if t["agent"] == "Doctor"),
        ""
    )

    evidence_vocab = extract_evidence_vocab(thoughts, summary_text)

    for t in thoughts:
        agent = t["agent"]
        text = t["content"]

        score = hallucination_score(
            agent=agent,
            text=text,
            doctor_text=doctor_text,
            summary_text=summary_text,
            evidence_vocab=evidence_vocab
        )

        scores[agent] = round(score, 3)

    return scores


def extract_evidence_vocab(thoughts, summary_text: str = ""):
    """
    Evidence vocabulary = Doctor outputs + approved Summary text
    """
    evidence_text = ""

    for t in thoughts:
        if t.get("agent") == "Doctor":
            evidence_text += " " + t.get("content", "")

    if summary_text:
        evidence_text += " " + summary_text

    return set(tokenize(evidence_text))




# ================================
# Claim-aware Hallucination Metric
# ================================

CLAIM_KEYWORDS = {
    "risk", "risks", "increase", "increases", "decrease", "decreases",
    "improve", "improves", "reduce", "reduces",
    "prevent", "prevents", "cause", "causes",
    "associated", "association", "linked", "leads",
    "evidence", "study", "studies", "clinical",
    "trial", "meta-analysis", "significant", "effect", "effects"
}


# ================================
# 12. Main interactive loop
# ================================
if __name__ == "__main__":
    console.print(Panel(
        "Multi-Agent Fitness Advisor\nFULL 272k PubMed + Real Citations + Groq LLMs + Summary Decision",
        title="READY", style="bold magenta"
    ))


    profile = input("\nEnter your profile (e.g., 42yo male, 88kg, bad knees, desk job): ")
    goal = input("What is your Goal? (e.g., lose fat and get strong without joint pain): ")

    console.print("\n[bold]Starting evidence-based debate with real PubMed studies...[/bold]\n")

    initial_state = {
        "user_question": goal,
        "user_profile": profile,
        "thoughts": [],
        "final_answer": "",
        "evidence_complete": False
    }

    # ✅ 1. Initialize final_state
    final_state = None

    # ✅ 2. Run the graph and capture ONLY coach output
    for step in app.stream(initial_state):
        for node_name, node_state in step.items():
            if node_name == "coach":
                final_state = node_state

    # ✅ 3. Generate PDF AFTER graph finishes
    if final_state and final_state.get("final_answer"):
        
        # =========================
        # Independent Hallucination Evaluation
        # =========================
        hallucination_scores = compute_independent_hallucination_scores(
            final_state["thoughts"],
            final_state.get("summary", "")
        )

        final_state.setdefault("eval", {})
        final_state["eval"]["hallucination_by_agent"] = hallucination_scores

        console.print("\n[bold red]Hallucination Scores for respective agents[/bold red]")
        for agent, score in hallucination_scores.items():
            console.print(f"{agent}: {score:.2f}")

        plan_text = final_state["final_answer"]
        
        role_scores = {}
        for thought in final_state["thoughts"]:
            agent = thought["agent"]
            content = thought["content"]
            role_scores[agent] = role_adherence_score(content, agent)

        overall_role_adherence = sum(role_scores.values()) / len(role_scores)
        
        console.print("\n[bold cyan]Role Adherence Scores[/bold cyan]")
        for role, score in role_scores.items():
            console.print(f"{role}: {score:.2f}")

        console.print(f"[bold]Overall Role Adherence:[/bold] {overall_role_adherence:.2f}")
        
        
        # =========================
        # Prompt Adherence Evaluation
        # =========================
        adherence_scores, drift_scores = compute_prompt_adherence_scores(
            final_state["thoughts"]
        )

        final_state.setdefault("eval", {})
        final_state["eval"]["prompt_adherence"] = adherence_scores
        final_state["eval"]["instruction_drift"] = drift_scores

        console.print("\n[bold green]Prompt Adherence Scores[/bold green]")
        for agent, score in adherence_scores.items():
            console.print(f"{agent}: {score:.2f}")

        console.print("\n[bold yellow]Instruction Drift Rates[/bold yellow]")
        for agent, score in drift_scores.items():
            console.print(f"{agent}: {score:.2f}")
            
        
        # =========================
        # Violation Rate Evaluation
        # =========================
        violation_scores = compute_violation_rates(final_state["thoughts"])

        final_state.setdefault("eval", {})
        final_state["eval"]["violation_rate_by_agent"] = violation_scores

        console.print("\n[bold red]Instruction Violation Rates[/bold red]")
        for agent, score in violation_scores.items():
            console.print(f"{agent}: {score:.3f}")

        
        os.makedirs("outputs/evaluation", exist_ok=True)
        with open("outputs/evaluation/role_adherence.json", "a") as f:
            json.dump(
                {
                    "role_scores": role_scores,
                    "overall": overall_role_adherence
                },
                f
            )
            f.write("\n")
        
        output_dir = "outputs/fitness_plans"
        os.makedirs(output_dir, exist_ok=True)

        pdf_name = "fitness_plan.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)

        generate_plan_pdf(pdf_path, profile, goal, plan_text)

        console.print(Panel(
            f"Done! Your plan has been saved as:\n{os.path.abspath(pdf_path)}",
            title="Complete",
            style="bold green"
        ))
    else:
        console.print("[red]No final plan generated – something went wrong in the graph.[/red]")
print("RUNNING FILE:", __file__)