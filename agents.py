# agents.py

def nutrition_agent(profile):
    goal = profile.get("goal", "")
    if goal == "Weight Loss":
        return "Protein-rich meals, calorie control, hydration reminders"
    elif goal == "Muscle Gain":
        return "High-protein meals, frequent snacks"
    else:
        return "Balanced nutrition and regular meals"


def exercise_agent(profile):
    # Default logic without activity
    return "20–30 min moderate activity (walking, stretching, light strength)"


def sleep_agent(profile):
    return "7–8 hours sleep, screen-free 1 hour before bed"
