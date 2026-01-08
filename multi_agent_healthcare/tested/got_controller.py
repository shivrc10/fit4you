#got_controller

from summariser_agent import summariser_agent
from critic_agent import critic_agent
from supporter_agent import supporter_agent

# Define edges for the Graph-of-Thoughts
edges = [
    ("Summary", "Critique"),
    ("Summary", "Support"),
    ("Critique", "Support")
]

def run_multi_agent_got(question, evidence):
    # Nodes = outputs of each agent
    nodes = {
        "Summary": summariser_agent(question, evidence),
        "Critique": critic_agent(question, evidence),
        "Support": supporter_agent(question, evidence)
    }

    # Graph traversal
    final_output = ""
    visited = set()

    def traverse(node):
        nonlocal final_output
        if node in visited:
            return
        final_output += f"\n--- {node} ---\n{nodes[node]}\n"
        visited.add(node)
        for edge in edges:
            if edge[0] == node:
                traverse(edge[1])

    traverse("Summary")
    return final_output

if __name__ == "__main__":
    question = "What are the benefits and risks of using metformin for type 2 diabetes?"
    evidence = (
        "Metformin is widely used for type 2 diabetes. It lowers blood sugar and reduces cardiovascular risk. "
        "Side effects may include gastrointestinal discomfort and rare cases of lactic acidosis."
    )

    result = run_multi_agent_got(question, evidence)
    print(result)
