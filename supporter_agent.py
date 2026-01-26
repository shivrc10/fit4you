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
# 1. Build / Load PubMed Vector DB
# ================================
DB_PATH = "./pubmed_272k_db"
os.makedirs(DB_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=DB_PATH)
COLLECTION_NAME = "pubmed_272k"

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
    console.print("[bold yellow]Building FULL 272k PubMedQA vector database (first run only)...[/bold yellow]")
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
                data = json.load(jf)
                all_data.update(data)
                console.print(f"[green]Loaded[/green] {f} → {len(data):,}")

    console.print(f"[bold blue]Indexing {len(all_data):,} PubMed papers[/bold blue]")

    batch_size = 64
    batch_texts, batch_ids, batch_docs = [], [], []

    for pid, item in tqdm(all_data.items(), desc="Embedding"):
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
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embedder.embed_documents(batch_texts),
                metadatas=batch_docs
            )
            batch_texts, batch_ids, batch_docs = [], [], []

    if batch_texts:
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embedder.embed_documents(batch_texts),
            metadatas=batch_docs
        )

    console.print("[bold green]PubMed database ready[/bold green]")
else:
    collection = client.get_collection(COLLECTION_NAME)
    console.print("[bold green]Loaded existing PubMed database[/bold green]")

# ================================
# 2. LLMs
# ================================
doctor_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
critic_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.7)
supporter_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.9)
coach_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
summarizer_llm = doctor_llm

def save_agent_message(agent, content, sources=None):
    cursor.execute(
        "INSERT INTO agent_messages (session_id, agent, content, sources) VALUES (%s,%s,%s,%s)",
        (SESSION_ID, agent, content, str(sources) if sources else None)
    )
    db.commit()

# ================================
# 3. Agent State
# ================================
class AgentState(TypedDict):
    user_question: str
    user_profile: str
    thoughts: List[Dict]
    final_answer: str
    evidence_complete: bool

# ================================
# 4. Doctor Node
# ================================
def doctor_node(state: AgentState):
    results = collection.query(
        query_texts=[state["user_question"] + " fitness exercise health impact"],
        n_results=12
    )

    context = "\n\n".join(
        f"[PMID:{results['ids'][0][i]}] {results['documents'][0][i][:1200]}"
        for i in range(len(results["documents"][0]))
    )

    prompt = f"""You are Doctor — evidence-based physician.
Use ONLY the studies below and cite as [PMID: x].
{context}
User: {state['user_profile']}
Goal: {state['user_question']}
"""

    resp = doctor_llm.invoke(prompt)
    save_agent_message("Doctor", resp.content, results["ids"][0][:4])

    return {
        **state,
        "thoughts": state["thoughts"] + [{
            "agent": "Doctor",
            "content": resp.content,
            "sources": results["ids"][0][:4]
        }]
    }

# ================================
# 5. Critic Node
# ================================
def critic_node(state: AgentState):
    resp = critic_llm.invoke(
        f"You are Critic. Critique:\n{state['thoughts'][-1]['content']}"
    )
    save_agent_message("Critic", resp.content)

    return {
        **state,
        "thoughts": state["thoughts"] + [{
            "agent": "Critic",
            "content": resp.content
        }]
    }

# ================================
# 6. Supporter Node
# ================================
def supporter_node(state: AgentState):
    resp = supporter_llm.invoke(
        f"You are Supporter.\nDebate:\n{json.dumps(state['thoughts'], indent=2)}"
    )
    save_agent_message("Supporter", resp.content)

    return {
        **state,
        "thoughts": state["thoughts"] + [{
            "agent": "Supporter",
            "content": resp.content
        }]
    }

# ================================
# 7. Summary Node (FIXED)
# ================================
def summary_node(state: AgentState):
    debate = "\n\n".join(
        f"{t['agent']}: {t['content'][:400]}" for t in state["thoughts"]
    )

    resp = summarizer_llm.invoke(
        f"Summarize in ≤150 words using bullets:\n{debate}"
    )

    console.print(
        Panel(resp.content, title="Evidence Summary", border_style="magenta")
    )

    while True:
        choice = input("Proceed to Coach? (y/n): ").strip().lower()
        if choice in ("y", "n"):
            break

    if choice == "y":
        return {
            **state,
            "evidence_complete": True
        }
    else:
        console.print("[yellow]Regenerating evidence debate...[/yellow]")
        return {
            **state,
            "thoughts": [],
            "evidence_complete": False
        }

# ================================
# 8. Coach Node
# ================================
def coach_node(state: AgentState):
    resp = coach_llm.invoke(
        f"You are Coach. Debate:\n{json.dumps(state['thoughts'], indent=2)}"
    )
    return {
        **state,
        "final_answer": resp.content,
        "thoughts": state["thoughts"] + [{
            "agent": "Coach",
            "content": resp.content
        }]
    }

# ================================
# 9. Graph
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

graph.add_conditional_edges(
    "summary",
    lambda s: "proceed" if s["evidence_complete"] else "regenerate",
    {"proceed": "coach", "regenerate": "doctor"}
)

graph.add_edge("coach", END)
app = graph.compile()

# ================================
# 10. Run
# ================================
if __name__ == "__main__":
    profile = input("Enter profile: ")
    goal = input("Enter goal: ")

    cursor.execute(
        "INSERT INTO sessions VALUES (%s,%s,%s)",
        (SESSION_ID, profile, goal)
    )
    db.commit()

    app.stream({
        "user_question": goal,
        "user_profile": profile,
        "thoughts": [],
        "final_answer": "",
        "evidence_complete": False
    })
