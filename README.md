# fit4you - AI-Powered Fitness Plan Generator
Fit4You is a local AI-powered backend service that generates personalized, knee-safe fitness plans using local LLMs via Ollama.
It supports plan generation, plan revision based on feedback, and safety-aware recommendations.

Features
Generate 2-week personalized fitness plans
Knee-friendly & safety-aware logic
Multiple plan alternatives (time-efficient, equipment-light)
Revise existing plans using expert/user feedback
Fully local LLM inference using Ollama
JSON-only responses (frontend-ready)

Tech Stack
Python 3.10
Flask
Ollama (local LLM runtime)
Models used:
llama3.2:latest (planner & summarizer)
gemma:2b (alternatives & safety)


Installation 
1. Install Ollama
2. Pull essentials models
3. Setup your environment
4. Run the Server
5. API call for generating the plan and reviceing it based on user feedback.
