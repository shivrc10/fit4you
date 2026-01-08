#new
# fitness_advisor_final_272k_pubmed_interactive.py
# 272k PubMedQA + Real Citations + Groq LLMs + Graph-of-Thoughts + Human-in-the-loop
# Author: Jaffar

import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, END
from typing import List, Dict, TypedDict
from rich.console import Console
from rich.panel import Panel
import chromadb
from tqdm import tqdm
from langchain_groq import ChatGroq

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
doctor_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

critic_llm    = ChatGroq(model="qwen/qwen3-32b", temperature=0.7)

supporter_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.9
)

coach_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

# ================================
# 3. Agent State
# ================================
class AgentState(TypedDict):
    user_question: str
    user_profile: str
    thoughts: List[Dict]
    final_answer: str

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

    while True:
        resp = doctor_llm.invoke(prompt)
        thought = {
            "id": len(state["thoughts"])+1,
            "agent": "Doctor",
            "content": resp.content,
            "sources": results['ids'][0][:4]
        }
        sources_str = " | ".join([f"[bold cyan]PMID:{pid}[/bold cyan]" for pid in results['ids'][0][:4]])
        console.print(Panel(resp.content + f"\n\nReal sources → {sources_str}",
                            title="[blue]Doctor – 272k Real PubMed[/blue]", border_style="blue"))
        
        choice = input("Do you want to regenerate Doctor output? (y/n): ").strip().lower()
        if choice != "y":
            break
    
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
    
    while True:
        resp = critic_llm.invoke(prompt)
        thought = {"id": len(state["thoughts"])+1, "agent": "Critic", "content": resp.content}
        console.print(Panel(resp.content, title="[red]Critic[/red]", border_style="red"))
        
        choice = input("Do you want to regenerate Critic output? (y/n): ").strip().lower()
        if choice != "y":
            break
    
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
    
    while True:
        resp = supporter_llm.invoke(prompt)
        thought = {"id": len(state["thoughts"])+1, "agent": "Supporter", "content": resp.content}
        console.print(Panel(resp.content, title="[green]Supporter[/green]", border_style="green"))
        
        choice = input("Do you want to regenerate Supporter output? (y/n): ").strip().lower()
        if choice != "y":
            break
    
    return {"thoughts": state["thoughts"] + [thought]}

# ================================
# 7. Coach Node
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

🧡 COACH’S ACTION PLAN
"""
    
    while True:
        resp = coach_llm.invoke(prompt)
        thought = {"id": len(state["thoughts"])+1, "agent": "Coach", "content": resp.content}
        console.print(Panel(resp.content, title="[bold yellow]Coach (YOU) – Final Plan[/bold yellow]", border_style="yellow"))
        
        choice = input("Do you want to regenerate Coach plan? (y/n): ").strip().lower()
        if choice != "y":
            break
    
    return {"thoughts": state["thoughts"] + [thought], "final_answer": resp.content}

# ================================
# 8. Build Graph-of-Thoughts
# ================================
graph = StateGraph(AgentState)
graph.add_node("doctor", doctor_node)
graph.add_node("critic", critic_node)
graph.add_node("supporter", supporter_node)
graph.add_node("coach", coach_node)

graph.set_entry_point("doctor")
graph.add_edge("doctor", "critic")
graph.add_edge("critic", "supporter")
graph.add_edge("supporter", "coach")
graph.add_edge("coach", END)
app = graph.compile()

# ================================
# 9. Main interactive loop
# ================================
if __name__ == "__main__":
    console.print(Panel("Multi-Agent Fitness Advisor\nFULL 272k PubMed + Real Citations + 3 LLMs + Interactive GOT", 
                        title="READY", style="bold magenta"))
    
    profile = input("\nEnter your profile ...(e.g., 42yo male, 88kg, bad knees, desk job): ")
    goal = input("What is your Goal? (e.g., lose fat and get strong without joint pain): ")
    
    console.print("\n[bold]Starting multiple thoughts with GOT approach and human-in-the-loop...[/bold]\n")
    
    for output in app.stream({
        "user_question": goal,
        "user_profile": profile,
        "thoughts": [],
        "final_answer": ""
    }):
        pass
    
    console.print(Panel("Done! Your bias-resistant, evidence-based plan is ready.", 
                        title="Complete", style="bold green"))
