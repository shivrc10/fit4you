from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

dataset = load_dataset("pubmed_qa", "pqa_labeled")
print(dataset)
# 1️⃣ Load model + tokenizer
#model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # or "meta-llama/Llama-3-8B-Instruct"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 2️⃣ Create a text-generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

# 3️⃣ Define your Summariser agent’s role prompt
SUMMARISER_PROMPT = """
You are a medical summariser.
Given a patient's question and retrieved medical evidence,
summarise the findings clearly and objectively.
Avoid giving medical advice — only summarise what the evidence says.
"""

# 4️⃣ Define a function to run the agent
def summariser_agent(question, evidence):
    prompt = f"{SUMMARISER_PROMPT}\n\nQuestion: {question}\n\nEvidence:\n{evidence}\n\nSummary:"
    output = generator(prompt)[0]["generated_text"]
    return output

# 5️⃣ Test the agent
if __name__ == "__main__":
    question = "What are the benefits of using metformin for type 2 diabetes?"
    evidence = (
        "Metformin is widely used as a first-line medication for type 2 diabetes. "
        "Studies show it lowers blood glucose levels and reduces the risk of cardiovascular complications. "
        "Side effects may include gastrointestinal discomfort."
    )
    result = summariser_agent(question, evidence)
    print("---- Summariser Agent Output ----")
    print(result)
