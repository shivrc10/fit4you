from datetime import datetime
import streamlit as st

def render_report(goal: str, structured_plan: dict):
    st.markdown("### 🎯 Goal")
    st.markdown(f"**{goal}**")
    st.caption(f"Generated on {datetime.now().strftime('%d %b %Y')}")
    st.divider()

    tabs = st.tabs(structured_plan.keys())

    for tab, (week, content) in zip(tabs, structured_plan.items()):
        with tab:
            st.markdown(f"#### {week}")
            st.markdown(content)
