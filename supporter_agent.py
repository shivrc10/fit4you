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
from uuid import uuid4
from db.connection import get_connection, create_tables

#init_db()
# Create tables if missing
create_tables()
db = get_connection()
cursor = db.cursor()
SESSION_ID = str(uuid4())  # Convert to string


console = Console()

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
   
    #files = ["pubmed272k/ori_pqac.json", "pubmed272k/ori_pqal.json", "pubmed272k/ori_pqau.json"]
    files = ["pubmed272k/ori_pqaa.json", "pubmed272k/or_pqal.json", "pubmed272k/ori_pqau.json"]

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

def save_agent_message(agent, content, sources=None):
    cursor.execute("""
        INSERT INTO agent_messages (session_id, agent, content, sources)
        VALUES (%s, %s, %s, %s)
    """, (SESSION_ID, agent, content, str(sources) if sources else None))
    db.commit()

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
        n_results=12
    )
   
    context_blocks = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        pmid = results['ids'][0][i]
        context_blocks.append(f"[PMID: {pmid}] {doc.strip()[:1200]}...")
   
    context = "\n\n".join(context_blocks)
   
    prompt = f"""You are Doctor — evidence-based physician using the FULL 272k PubMedQA dataset.
Use ONLY the real studies below. Cite them as [PMID: 12345678].
Real evidence retrieved:
{context}
User: {state['user_profile']}
Goal: {state['user_question']}
Give only medical facts, safety profile, and evidence level.
Never give training plans.
Output exactly:
DOCTOR: [your response with real PMID citations]"""
   
    resp = doctor_llm.invoke(prompt)
    sources_str = " | ".join([f"[bold cyan]PMID:{pid}[/bold cyan]" for pid in results['ids'][0][:4]])
    console.print(Panel(resp.content + f"\n\nReal sources → {sources_str}",
                        title="[blue]Doctor – 272k Real PubMed[/blue]", border_style="blue"))
   
    thought = {
        "id": len(state["thoughts"])+1,
        "agent": "Doctor",
        "content": resp.content,
        "sources": results['ids'][0][:4]
    }
   
    save_agent_message("Doctor", resp.content, results['ids'][0][:4])
    return {"thoughts": state["thoughts"] + [thought]}

# ================================
# 5. Critic Node
# ================================
def critic_node(state: AgentState):
    prev = state["thoughts"][-1]["content"]
    prompt = f"""You are Critic. Attack weak evidence, small samples, bias, or overgeneralization.
Doctor claimed (with real PMIDs):
{prev}
Tear it apart if needed. Be ruthless but fair.
CRITIC: """
   
    resp = critic_llm.invoke(prompt)
    console.print(Panel(resp.content, title="[red]Critic[/red]", border_style="red"))
   
    thought = {"id": len(state["thoughts"])+1, "agent": "Critic", "content": resp.content}
    save_agent_message("Critic", resp.content)
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
    save_agent_message("Supporter", resp.content)
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
        choice = input("\nDo you agree with this summary and want to proceed to Coach’s personalized plan? (y/n): ").strip().lower()
        if choice in ["y", "n"]:
            break
   
    if choice == "y":
        return {"evidence_complete": True}
    else:
        console.print("[yellow]Regenerating evidence debate...[/yellow]\n")
        return {"thoughts": [], "evidence_complete": False}  # Reset for regeneration

# ================================
# 8. Coach Node
# ================================
def coach_node(state: AgentState):
    debate = json.dumps([{"agent": t["agent"], "text": t["content"][:1000]} for t in state["thoughts"]], indent=2)
    prompt = f"""You are Coach — the practical expert.
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
COACH’S ACTION PLAN
"""
   
    resp = coach_llm.invoke(prompt)
    console.print(Panel(resp.content, title="[bold yellow]Coach (YOU) – Final Plan[/bold yellow]", border_style="yellow"))
   
    thought = {"id": len(state["thoughts"])+1, "agent": "Coach", "content": resp.content}
    return {"thoughts": state["thoughts"] + [thought], "final_answer": resp.content}

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
# 10. Main interactive loop
# ================================
if __name__ == "__main__":
    console.print(Panel("Multi-Agent Fitness Advisor\nFULL 272k PubMed + Real Citations + Groq LLMs + Summary Decision",
                        title="READY", style="bold magenta"))
   
    profile = input("\nEnter your profile (e.g., 42yo male, 88kg, bad knees, desk job): ")
    goal = input("What is your Goal? (e.g., lose fat and get strong without joint pain): ")
    cursor.execute("""INSERT INTO sessions (session_id, user_profile, user_goal) VALUES (%s, %s, %s)""", (SESSION_ID, profile, goal))
    db.commit()

    print(f"Session Started → {SESSION_ID}")

    console.print("\n[bold]Starting evidence-based debate with real PubMed studies...[/bold]\n")
   
    initial_state = {
        "user_question": goal,
        "user_profile": profile,
        "thoughts": [],
        "final_answer": "",
        "evidence_complete": False
    }
   
    for output in app.stream(initial_state):
        pass
   
    console.print(Panel("Done! Your evidence-based, personalized fitness plan is ready.",
                        title="Complete", style="bold green"))