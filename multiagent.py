# ============================================================
# multiagent.py
# Multi-Agent Fitness Advisor (Frontend + CLI Compatible)
# ============================================================

import json
import os
from typing import List, Dict, TypedDict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
import chromadb
from langchain_groq import ChatGroq

# -------------------------
# CONFIG
# -------------------------
DEV_MODE = True
DB_PATH = "./pubmed_272k_db"
COLLECTION_NAME = "pubmed_272k"

# -------------------------
# Vector DB
# -------------------------
os.makedirs(DB_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=DB_PATH)

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

existing = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing:
    collection = client.get_collection(COLLECTION_NAME)
else:
    collection = client.create_collection(COLLECTION_NAME)

# -------------------------
# LLMs
# -------------------------
doctor_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
critic_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.7)
supporter_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.9)
coach_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
summarizer_llm = doctor_llm

# -------------------------
# State
# -------------------------
class AgentState(TypedDict):
    user_question: str
    user_profile: str
    thoughts: List[Dict]
    final_answer: str
    evidence_complete: bool

# -------------------------
# Doctor
# -------------------------
def doctor_node(state: AgentState):
    results = collection.query(
        query_texts=[state["user_question"] + " fitness exercise health impact"],
        n_results=6
    )

    context = []
    for i in range(len(results["documents"][0])):
        pmid = results["ids"][0][i]
        doc = results["documents"][0][i]
        context.append(f"[PMID:{pmid}] {doc[:800]}")

    prompt = f"""
You are Doctor.
Use ONLY real PubMed evidence.
Medical facts & safety only.

Evidence:
{chr(10).join(context)}

Profile: {state['user_profile']}
Goal: {state['user_question']}

DOCTOR:
"""
    resp = doctor_llm.invoke(prompt)

    return {
        "thoughts": state["thoughts"] + [{
            "agent": "Doctor",
            "content": resp.content,
            "sources": results["ids"][0][:4]
        }]
    }

# -------------------------
# Critic
# -------------------------
def critic_node(state: AgentState):
    prompt = f"""
You are Critic.
Attack weak evidence, bias, or overclaims.

Doctor said:
{state['thoughts'][-1]['content']}

CRITIC:
"""
    resp = critic_llm.invoke(prompt)

    return {
        "thoughts": state["thoughts"] + [{
            "agent": "Critic",
            "content": resp.content
        }]
    }

# -------------------------
# Supporter
# -------------------------
def supporter_node(state: AgentState):
    prompt = f"""
You are Supporter.
Highlight realistic benefits and encouragement.

Profile: {state['user_profile']}
Goal: {state['user_question']}

Debate:
{json.dumps(state['thoughts'], indent=2)}

SUPPORTER:
"""
    resp = supporter_llm.invoke(prompt)

    return {
        "thoughts": state["thoughts"] + [{
            "agent": "Supporter",
            "content": resp.content
        }]
    }

# -------------------------
# Summary (Frontend-safe)
# -------------------------
def summary_node(state: AgentState):
    prompt = f"""
Summarize the debate in under 150 words.
One bullet per agent.
Neutral tone.
"""
    resp = summarizer_llm.invoke(prompt)

    return {
        "evidence_complete": True,
        "thoughts": state["thoughts"] + [{
            "agent": "Summary",
            "content": resp.content
        }]
    }

# -------------------------
# Coach (TEXT OUTPUT – SAFE)
# -------------------------
def coach_node(state: AgentState):
    prompt = f"""
You are Coach.
Give a practical, detailed 4–8 week fitness plan.
Do NOT repeat medical discussion.

Profile: {state['user_profile']}
Goal: {state['user_question']}

Debate:
{json.dumps(state['thoughts'], indent=2)}

COACH:
"""
    resp = coach_llm.invoke(prompt)

    return {
        "final_answer": resp.content,
        "thoughts": state["thoughts"] + [{
            "agent": "Coach",
            "content": resp.content
        }]
    }

# -------------------------
# Graph
# -------------------------
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
graph.add_edge("summary", "coach")
graph.add_edge("coach", END)

app = graph.compile()

# ============================================================
# FRONTEND ENTRY POINT
# ============================================================
def run_pipeline(profile: str, goal: str) -> dict:
    """
    Streamlit-safe callable
    """
    initial_state = {
        "user_question": goal,
        "user_profile": profile,
        "thoughts": [],
        "final_answer": "",
        "evidence_complete": True
    }

    final_state = None
    for step in app.stream(initial_state):
        for _, state in step.items():
            final_state = state

    return {
        "metadata": {
            "profile": profile,
            "goal": goal
        },
        "plan_text": {
            "raw_plan": final_state.get("final_answer", "")
        },
        "agents": final_state.get("thoughts", [])
    }

# ============================================================
# CLI MODE (OPTIONAL)
# ============================================================
# if __name__ == "__main__":
#     profile = input("Profile: ")
#     goal = input("Goal: ")

#     result = run_pipeline(profile, goal)
#     print("\n===== FINAL PLAN =====\n")
#     print(result["plan_text"]["raw_plan"])
