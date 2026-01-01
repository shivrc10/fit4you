# planner.py

from agents import nutrition_agent, exercise_agent, sleep_agent

def generate_plan(profile):
    return {
        "score": 92,
        "morning": nutrition_agent(profile),
        "afternoon": exercise_agent(profile),
        "evening": sleep_agent(profile)
    }
