from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

CRITIC_PROMPT = """
You are a skeptical medical critic.
Given a patient's question and medical evidence, highlight potential weaknesses, risks, or gaps.
Point out side effects, study limitations, or situations where the treatment might not be suitable.
Do not give personal advice; critique objectively.
"""

def critic_agent(question, evidence):
    prompt = f"{CRITIC_PROMPT}\n\nQuestion: {question}\nEvidence:\n{evidence}\nCritique:"
    output = generator(prompt)[0]["generated_text"]
    return output
