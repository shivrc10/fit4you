# ============================================================
# Interactive Graph-of-Thoughts Fitness Advisor
# REAL PubMedQA (272k) + Human-in-the-Loop Expert Evaluation
# ============================================================

import json
import os
from typing import List, Dict, TypedDict

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, END
import chromadb

from rich.console import Console
from rich.panel import Panel

console = Console()

# ============================================================
# 1. Load PubMed Vector Database (already built)
# ============================================================

DB_PATH = "./pubmed_272k_db"
COLLECTION_NAME = "pubmed_272k"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

embedder = OllamaEmbeddings(model="mxbai-embed-large")

console.print("[bold green]Loaded 272k PubMed vector database[/bold green]")

# ============================================================
# 2. Models (smaller + faster where possible)
# ============================================================

#doctor_llm  = ChatOllama(model="phi3:mini", temperature=0.2)
doctor_llm  = ChatOllama(model="phi3", temperature=0.2)
review_llm  = ChatOllama(model="gemma2:9b", temperature=0.6)
coach_llm   = ChatOllama(model="llama3.1:8b", temperature=0.5)

# ============================================================
# 3. State Definition
# ============================================================

class AgentState(TypedDict):
    user_question: str
    user_profile: str
    thoughts: List[Dict]
    final_answer: str
    expert_decision: str
    expert_feedback: str

# ============================================================
# 4. Doctor Node (Evidence ONLY)
# ============================================================

def doctor_node(state: AgentState):
    results = collection.query(
        query_texts=[state["user_question"] + " exercise fitness health"],
        n_results=5
    )

    context_blocks = []
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        pmid = results["ids"][0][i]
        context_blocks.append(
            f"[PMID:{pmid}] {doc.strip()[:400]}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are Doctor — evidence-based physician.

Use ONLY the real PubMed evidence below.
Do NOT speculate.
Cite papers as [PMID:XXXX].

EVIDENCE:
{context}

User profile:
{state["user_profile"]}

Goal:
{state["user_question"]}

Output exactly:

DOCTOR:
- Key findings
- Safety considerations
- Evidence strength
"""

    resp = doctor_llm.invoke(prompt)

    console.print(
        Panel(resp.content, title="[blue]Doctor – PubMed Evidence[/blue]")
    )

    return {
        "thoughts": state["thoughts"] + [{
            "agent": "Doctor",
            "content": resp.content,
            "sources": results["ids"][0]
        }]
    }

# ============================================================
# 5. Reviewer Node (Critic + Supporter)
# ============================================================

def reviewer_node(state: AgentState):
    prev = state["thoughts"][-1]["content"]

    prompt = f"""
You are Reviewer.

TASKS:
1. Critique weak evidence, bias, limitations.
2. Highlight realistic benefits and strengths.

Doctor said:
{prev}

Output EXACTLY:

CRITIQUE:
- ...

STRENGTHS:
- ...
"""

    resp = review_llm.invoke(prompt)

    console.print(
        Panel(resp.content, title="[purple]Reviewer (Critic + Supporter)[/purple]")
    )

    return {
        "thoughts": state["thoughts"] + [{
            "agent": "Reviewer",
            "content": resp.content
        }]
    }

# ============================================================
# 6. Human / Expert Evaluation Node (INTERACTIVE)
# ============================================================

def human_evaluator_node(state: AgentState):
    console.print("\n[bold cyan]EXPERT EVALUATION STEP[/bold cyan]")
    console.print("1 → Approve and generate plan")
    console.print("2 → Refine plan with feedback")
    console.print("3 → Reject and rerun Doctor\n")

    choice = input("Your choice (1/2/3): ").strip()
    feedback = ""

    if choice in ["2", "3"]:
        feedback = input("Enter expert feedback / constraints: ")

    console.print(
        Panel(
            f"Decision: {choice}\nFeedback: {feedback}",
            title="[cyan]Expert Decision[/cyan]"
        )
    )

    return {
        "thoughts": state["thoughts"] + [{
            "agent": "HumanEvaluator",
            "content": f"Choice={choice} | Feedback={feedback}"
        }],
        "expert_decision": choice,
        "expert_feedback": feedback
    }

# ============================================================
# 7. Coach Node (Action Plan ONLY)
# ============================================================

def coach_node(state: AgentState):
    debate = json.dumps(
        [{"agent": t["agent"], "text": t["content"][:700]}
         for t in state["thoughts"]],
        indent=2
    )

    expert_feedback = state.get("expert_feedback", "")

    prompt = f"""
You are Coach — practical fitness expert.

Debate so far:
{debate}

Expert feedback (must obey strictly):
{expert_feedback}

RULES:
- Do NOT repeat medical facts
- Be concrete and practical
- Personalize to user
- Assume safety constraints already handled

Deliver:
• 4–8 week weekly schedule table
• Progression rules
• Injury modifications
• Nutrition timing
• Habit system
• Ask exactly 2 follow-up questions

FINAL ACTION PLAN:
"""

    resp = coach_llm.invoke(prompt)

    console.print(
        Panel(resp.content, title="[bold yellow]Coach – Final Plan[/bold yellow]")
    )

    return {
        "thoughts": state["thoughts"] + [{
            "agent": "Coach",
            "content": resp.content
        }],
        "final_answer": resp.content
    }

# ============================================================
# 8. Conditional Routing Logic
# ============================================================

def expert_router(state: AgentState):
    if state.get("expert_decision") == "3":
        return "doctor"
    return "coach"

# ============================================================
# 9. Build Graph
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("doctor", doctor_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("human_eval", human_evaluator_node)
graph.add_node("coach", coach_node)

graph.set_entry_point("doctor")

graph.add_edge("doctor", "reviewer")
graph.add_edge("reviewer", "human_eval")

graph.add_conditional_edges(
    "human_eval",
    expert_router,
    {
        "doctor": "doctor",
        "coach": "coach"
    }
)

graph.add_edge("coach", END)

app = graph.compile()

# ============================================================
# 10. Run
# ============================================================

if __name__ == "__main__":
    console.print(
        Panel(
            "Interactive Graph-of-Thoughts Fitness Advisor\n"
            "REAL PubMed + Human-in-the-Loop",
            title="READY",
            style="bold magenta"
        )
    )

    profile = input("\nEnter your profile (e.g., 42yo male, knee pain): ")
    goal = input("Enter your goal (e.g., lose fat safely): ")

    console.print("\n[bold]Starting reasoning pipeline...[/bold]\n")

    for _ in app.stream({
        "user_question": goal,
        "user_profile": profile,
        "thoughts": [],
        "final_answer": "",
        "expert_decision": "",
        "expert_feedback": ""
    }):
        pass

    console.print(
        Panel(
            "Completed.\nBias-resistant, evidence-based, expert-approved plan generated.",
            title="DONE",
            style="bold green"
        )
    )
