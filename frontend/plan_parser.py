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
