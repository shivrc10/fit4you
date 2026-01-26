# multiagent.py
# Streamlit-safe multi-agent backend: Doctor → Critic → Supporter → Summary → Coach
import json
from typing import TypedDict, List, Dict

import chromadb
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===================================================
# Vector DB (load existing 272k PubMed DB)
# ===================================================
client = chromadb.PersistentClient(path="./pubmed_272k_db")
collection = client.get_collection("pubmed_272k")

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ===================================================
# LLMs (Groq)
# ===================================================
doctor_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
critic_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.7)
supporter_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.9)
summarizer_llm = doctor_llm
coach_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

# ===================================================
# State definition
# ===================================================
class AgentState(TypedDict):
    user_profile: str
    user_question: str
    thoughts: List[Dict]
    final_answer: str
    stop_after_summary: bool  # controls whether to go to Coach

# ===================================================
# Nodes
# ===================================================
def doctor_node(state: AgentState):
    results = collection.query(
        query_texts=[state["user_question"] + " fitness exercise health impact"],
        n_results=12,
    )

    context = "\n\n".join(
        f"[PMID:{results['ids'][0][i]}] {results['documents'][0][i][:1200]}"
        for i in range(len(results["documents"][0]))
    )

    prompt = f"""
You are Doctor — an evidence-based physician using a 272k PubMedQA database.
Use ONLY the studies below. Cite them as [PMID: 12345678].
Give medical facts, safety profile, and evidence level.
Never give training plans.

EVIDENCE:
{context}

USER PROFILE:
{state["user_profile"]}

GOAL:
{state["user_question"]}

OUTPUT:
DOCTOR: [your response with real PMID citations]
"""
    resp = doctor_llm.invoke(prompt)

    return {
        **state,
        "thoughts": state["thoughts"] + [{
            "agent": "Doctor",
            "content": resp.content,
            "sources": results["ids"][0][:4],
        }]
    }


def critic_node(state: AgentState):
    prev = state["thoughts"][-1]["content"]
    prompt = f"""You are Critic. Attack weak evidence, bias, small samples, or overgeneralization.
Doctor claimed (with real PMIDs):
{prev}
Be ruthless but fair.

CRITIC:"""
    resp = critic_llm.invoke(prompt)

    return {
        **state,
        "thoughts": state["thoughts"] + [{
            "agent": "Critic",
            "content": resp.content,
        }]
    }


def supporter_node(state: AgentState):
    debate_snippets = json.dumps(
        [t["content"][:500] for t in state["thoughts"]],
        indent=2
    )

    prompt = f"""You are Supporter. Highlight realistic benefits, success patterns, and hope.
User profile: {state['user_profile']}
Goal: {state['user_question']}

Previous debate:
{debate_snippets}

Be encouraging and realistic.

SUPPORTER:"""
    resp = supporter_llm.invoke(prompt)

    return {
        **state,
        "thoughts": state["thoughts"] + [{
            "agent": "Supporter",
            "content": resp.content,
        }]
    }


def summary_node(state: AgentState):
    debate = "\n\n".join(
        f"{t['agent']}: {t['content'][:400]}"
        for t in state["thoughts"]
    )

    prompt = f"""Summarize the full evidence debate in 150 words or less.
Use bullet points (one per agent).
Be neutral, factual, and concise.
Highlight key pros/cons, evidence strength, and implications for the user's goal.

DEBATE:
{debate}

OUTPUT:
Only the summary bullets, no intro or outro.
"""
    resp = summarizer_llm.invoke(prompt)

    return {
        **state,
        "thoughts": state["thoughts"] + [{
            "agent": "Summary",
            "content": resp.content.strip(),
        }]
    }


def coach_node(state: AgentState):
    # Only called when stop_after_summary == False
    debate = json.dumps(
        [{"agent": t["agent"], "text": t["content"][:1000]} for t in state["thoughts"]],
        indent=2
    )

    prompt = f"""You are Coach — the practical expert.

RULES:
- Do NOT repeat medical evidence or risks.
- Do NOT critique evidence.
- ONLY provide a concrete 4–8 week plan.

INCLUDE (MANDATORY):
• A weekly schedule for 4–8 weeks.
• For each week: days, exact exercises, sets, reps, and intensities.
• Progression rules across weeks.
• Modifications for injuries or joint pain if relevant.
• Nutrition timing guidelines.
• A simple habit system.
• 2 reflective questions at the end.

FORMAT:
Return Markdown with:
- A heading "COACH ACTION PLAN"
- Clear "Week 1", "Week 2", ... labels.

USER PROFILE:
{state["user_profile"]}

GOAL:
{state["user_question"]}

DEBATE CONTEXT (for your reference):
{debate}
"""
    resp = coach_llm.invoke(prompt)

    return {
        **state,
        "final_answer": resp.content,
        "thoughts": state["thoughts"] + [{
            "agent": "Coach",
            "content": resp.content,
        }]
    }

# ===================================================
# Graph
# ===================================================
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

# If stop_after_summary=True → end after summary.
# If False → go to coach.
graph.add_conditional_edges(
    "summary",
    lambda s: END if s["stop_after_summary"] else "coach",
    {
        END: END,
        "coach": "coach",
    },
)

graph.add_edge("coach", END)

app = graph.compile()

# ===================================================
# Public API used by Streamlit
# ===================================================
def run_pipeline(profile: str, goal: str, proceed: bool = False, regenerate: bool = False):
    """
    First call (proceed=False):
      - runs Doctor → Critic → Supporter → Summary
      - returns agents + awaiting_confirmation=True

    Second call (proceed=True):
      - runs full graph again with stop_after_summary=False
      - returns agents + plan_text + can_proceed=True
    """

    state: AgentState = {
        "user_profile": profile,
        "user_question": goal,
        "thoughts": [],
        "final_answer": "",
        "stop_after_summary": not proceed,
    }

    final_state = None
    for step in app.stream(state):
        for _, value in step.items():
            final_state = value

    if final_state is None:
        raise RuntimeError("Pipeline produced no output")

    thoughts = final_state["thoughts"]

    if not proceed:
        # evidence-only run (stop at Summary)
        return {
            "agents": thoughts,
            "awaiting_confirmation": True,
            "can_proceed": False,
            "regenerated": regenerate,
        }

    # coach run: final_answer should be filled
    return {
        "agents": thoughts,
        "awaiting_confirmation": False,
        "can_proceed": True,
        "plan_text": {
            "raw_plan": final_state["final_answer"],
        },
    }
