# fitness_advisor_final_272k_pubmed_interactive_summary.py
# 272k PubMedQA + Real Citations + Groq LLMs + Graph-of-Thoughts

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
from uuid import uuid4
from db.connection import get_connection, create_tables

# ================================
# DB INIT
# ================================
create_tables()
db = get_connection()
cursor = db.cursor()
SESSION_ID = str(uuid4())

console = Console()

# ================================
# VECTOR DB
# ================================
DB_PATH = "./pubmed_272k_db"
os.makedirs(DB_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=DB_PATH)
COLLECTION_NAME = "pubmed_272k"

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
    console.print("[bold yellow]Building PubMed vector DB...[/bold yellow]")
    collection = client.create_collection(COLLECTION_NAME)

    files = [
        "pubmed272k/ori_pqaa.json",
        "pubmed272k/or_pqal.json",
        "pubmed272k/ori_pqau.json"
    ]

    all_data = {}
    for f in files:
        if os.path.exists(f):
            with open(f) as jf:
                all_data.update(json.load(jf))

    batch_texts, batch_ids, batch_docs = [], [], []
    for pid, item in tqdm(all_data.items()):
        text = (
            f"Question: {item.get('QUESTION','')}\n"
            f"Context: {' '.join(item.get('CONTEXTS', []))}\n"
            f"Answer: {item.get('LONG_ANSWER','')}\n"
            f"Conclusion: {item.get('final_decision','')}"
        )
        batch_texts.append(text)
        batch_ids.append(pid)
        batch_docs.append({"question": item.get("QUESTION","")[:200]})

        if len(batch_texts) >= 64:
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embedder.embed_documents(batch_texts),
                metadatas=batch_docs
            )
            batch_texts, batch_ids, batch_docs = [], [], []

else:
    collection = client.get_collection(COLLECTION_NAME)
    console.print("[bold green]Loaded existing PubMed DB[/bold green]")

# ================================
# LLMs
# ================================
doctor_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
critic_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.7)
supporter_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.9)
coach_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
summarizer_llm = doctor_llm

# ================================
# DB LOGGING
# ================================
def save_agent_message(agent, content, sources=None):
    cursor.execute(
        """
        INSERT INTO agent_messages (session_id, agent, content, sources)
        VALUES (%s, %s, %s, %s)
        """,
        (SESSION_ID, agent, content, str(sources) if sources else None)
    )
    db.commit()

# ================================
# PROGRESS EMITTER
# ================================
def emit_progress(state, agent_name):
    state["progress_agent"] = agent_name
    return state

# ================================
# AGENT STATE  ✅ FIXED
# ================================
class AgentState(TypedDict):
    user_question: str
    user_profile: str
    thoughts: List[Dict]
    final_answer: str
    evidence_complete: bool
    progress_agent: str   # ✅ ADDED

# ================================
# DOCTOR
# ================================
def doctor_node(state: AgentState):
    state = emit_progress(state, "Doctor")

    results = collection.query(
        query_texts=[state["user_question"] + " fitness exercise health"],
        n_results=12
    )

    context = "\n\n".join(
        f"[PMID:{pid}] {doc[:1200]}"
        for pid, doc in zip(results["ids"][0], results["documents"][0])
    )

    resp = doctor_llm.invoke(f"""
    You are Doctor. Use ONLY PubMed evidence.
    {context}
    """)

    save_agent_message("Doctor", resp.content, results["ids"][0][:4])

    return {
        "thoughts": state["thoughts"] + [{
            "agent": "Doctor",
            "content": resp.content,
            "sources": results["ids"][0][:4]
        }],
        "progress_agent": state["progress_agent"]  # ✅ REQUIRED
    }

# ================================
# CRITIC
# ================================
def critic_node(state: AgentState):
    state = emit_progress(state, "Critic")

    resp = critic_llm.invoke(state["thoughts"][-1]["content"])
    save_agent_message("Critic", resp.content)

    return {
        "thoughts": state["thoughts"] + [{
            "agent": "Critic",
            "content": resp.content
        }],
        "progress_agent": state["progress_agent"]
    }

# ================================
# SUPPORTER
# ================================
def supporter_node(state: AgentState):
    state = emit_progress(state, "Supporter")

    resp = supporter_llm.invoke(json.dumps(state["thoughts"][-2:], indent=2))
    save_agent_message("Supporter", resp.content)

    return {
        "thoughts": state["thoughts"] + [{
            "agent": "Supporter",
            "content": resp.content
        }],
        "progress_agent": state["progress_agent"]
    }

# ================================
# SUMMARY
# ================================
def summary_node(state: AgentState):
    resp = summarizer_llm.invoke("Summarize debate in <150 words.")
    return {
        "evidence_complete": True,
        "progress_agent": state.get("progress_agent", "Summary")
    }

# ================================
# COACH
# ================================
def coach_node(state: AgentState):
    state = emit_progress(state, "Coach")

    resp = coach_llm.invoke(json.dumps(state["thoughts"], indent=2))
    save_agent_message("Coach", resp.content)

    return {
        "final_answer": resp.content,
        "thoughts": state["thoughts"] + [{
            "agent": "Coach",
            "content": resp.content
        }],
        "progress_agent": state["progress_agent"]
    }

# ================================
# GRAPH
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
graph.add_edge("summary", "coach")
graph.add_edge("coach", END)

app = graph.compile()
