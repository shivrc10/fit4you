from datetime import datetime
import streamlit as st
import re

def clean_llm_text(text: str) -> str:
    # Remove bold / italic markers
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Remove stray heading markers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    return text.strip()

def sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]

def render_report(goal: str, structured_plan: dict):
    st.markdown("### 🎯 Goal")
    st.markdown(f"**Your goal:** {sentence_case(goal)}")
    st.caption(f"Generated on {datetime.now().strftime('%d %b %Y')}")
    st.divider()

    tabs = st.tabs(structured_plan.keys())

    for tab, (week, content) in zip(tabs, structured_plan.items()):
        with tab:
            st.subheader(week)

            clean_content = clean_llm_text(content)

            st.markdown(clean_content)
