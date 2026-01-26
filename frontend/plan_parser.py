import re

WEEK_REGEX = re.compile(
    r"(Week[s]?\s*\d+(\s*[-–]\s*\d+)?)[\s:–-]*",
    re.IGNORECASE
)

def parse_plan_into_weeks(raw_text: str) -> dict:
    matches = list(WEEK_REGEX.finditer(raw_text))
    if not matches:
        return {"Plan": raw_text.strip()}

    structured = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        title = match.group(1).title()
        content = raw_text[start:end].strip()
        structured[title] = content

    return structured

# report_layout.py
# Renders structured weekly plan as beautiful cards

# plan_parser.py
# Parses raw Coach Markdown into structured weekly dicts for rendering/PDF

# import re
# from typing import Dict, List  # ← ADD THIS LINE

# def parse_plan_into_weeks(raw_plan: str) -> Dict[str, str]:
#     """
#     Parses Coach's raw Markdown output into {"Week 1": "content...", "Week 2": "..."}
    
#     Expects Coach to use clear "Week X" headings (as enforced in prompt).
#     """
#     if not raw_plan:
#         return {}
    
#     # Split on Week headings (case-insensitive, flexible patterns)
#     weeks_pattern = r'(?i)(?:^|\n)(week\s+(\d+)(?:\s*:?\s*|$))'
#     weeks = re.split(weeks_pattern, raw_plan.strip())
    
#     structured = {}
#     current_week = None
    
#     # Reconstruct weeks from split
#     for i in range(1, len(weeks), 3):
#         week_num = weeks[i+1].strip() if i+1 < len(weeks) else ""
#         week_content = weeks[i+2].strip() if i+2 < len(weeks) else ""
        
#         if week_num:
#             current_week = f"Week {week_num}"
#             structured[current_week] = week_content
#         elif current_week and week_content:
#             # Append to current week
#             structured[current_week] += "\n" + week_content
    
#     # Fallback: if no weeks found, treat as single block
#     if not structured:
#         structured["Plan Overview"] = raw_plan
    
#     return structured
