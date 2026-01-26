import streamlit as st
import sys
import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multiagent import run_pipeline
from plan_parser import parse_plan_into_weeks
from report_layout import render_report

# ===================================================
# PAGE CONFIG - MUST BE FIRST
# ===================================================
st.set_page_config(
    page_title="FIT4YOU – My Plan", layout="wide", initial_sidebar_state="collapsed"
)

# ===================================================
# LOAD CSS FIRST
# ===================================================
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass


logo_col, spacer, nav1, nav2, nav3 = st.columns([2.2, 6.0, 1.0, 1.0, 1.0])

with logo_col:
    st.markdown(
        """
        <div class="app-logo">
            <span class="logo-icon">🩺</span>
            <span class="logo-text">FIT4YOU</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Where multiple perspectives shape safer fitness decisions")

with nav1:
    st.page_link("app.py", label="My Plan")

with nav2:
    st.page_link("pages/work.py", label="How it works")

with nav3:
    st.page_link("pages/about.py", label="About")

st.divider()


# ===================================================
# SESSION STATE
# ===================================================
DEFAULT_STATE = {
    "loading": False,
    "progress": 0,
    "result": None,
    "profile": "",
    "goal": "",
    "selected_agent": None,
}

for k, v in DEFAULT_STATE.items():
    st.session_state.setdefault(k, v)


def reset_app():
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = v
    st.rerun()


# ===================================================
# AGENT DIALOG
# ===================================================


@st.dialog("Agent details")
def show_agent_dialog(agent_name: str):
    explanations = {
        "Doctor": "Reviews real PubMed evidence and identifies medical risks.",
        "Critic": "Challenges weak evidence and highlights limitations.",
        "Supporter": "Balances critique with realistic encouragement.",
        # "Coach": "Creates your final personalized action plan.",
        "Summary": "Condenses the multi-agent debate into a neutral overview.",
    }

    st.markdown(f"## {agent_name}")
    st.caption(explanations.get(agent_name, ""))

    agent_data = st.session_state.agent_outputs.get(agent_name)

    st.divider()

    if agent_data:
        st.markdown("### Agent contribution")
        st.markdown(agent_data.get("content", "_No content available._"))

        if agent_name == "Doctor" and agent_data.get("sources"):
            st.divider()
            st.markdown("### Evidence sources")
            for pmid in agent_data["sources"]:
                st.markdown(f"- `{pmid}`")
    else:
        st.warning("No output available for this agent.")

    st.divider()

    if st.button("Close", use_container_width=True):
        st.session_state.selected_agent = None
        st.rerun()


# ===================================================
# PDF GENERATOR
# ===================================================
def generate_pdf(goal: str, structured_plan: dict) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    x, y = 50, height - 50

    c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, "Health Report")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(x, y, f"Goal: {goal}")
    y -= 20

    for week, content in structured_plan.items():
        c.setFont("Helvetica-Bold", 13)
        c.drawString(x, y, week)
        y -= 18

        c.setFont("Helvetica", 11)
        for line in content.split("\n"):
            for wrapped in wrap(line, 90):
                if y < 60:
                    c.showPage()
                    y = height - 50
                c.drawString(x, y, wrapped)
                y -= 14
        y -= 10

    c.save()
    buffer.seek(0)
    return buffer.read()


# ===================================================
# HEADER
# # ===================================================
# st.markdown("## 🩺 FIT4YOU")
# st.caption("Where multiple perspectives shape safer fitness decisions")
# st.divider()

# ===================================================
# LAYOUT
# ===================================================
left, right = st.columns([1, 2.2], gap="large")

# LEFT COLUMN
with left:
    with st.container(key="card-profile"):
        st.subheader("👤 Profile & Goal")

        st.session_state.profile = st.text_area(
            "Profile",
            value=st.session_state.profile,
            height=120,
        )

        st.session_state.goal = st.text_input(
            "Goal",
            value=st.session_state.goal,
        )

        if st.button("Generate Plan", type="primary"):
            st.session_state.loading = True
            st.session_state.progress = 0
            st.session_state.result = None
            st.session_state.selected_agent = None
            st.rerun()

    with st.container(key="card-agents"):
        h_l, h_r = st.columns([6, 1])

        with h_l:
            st.subheader("Agents Working")
            st.caption("ℹ️ Click agent icon to view its contribution")

        with h_r:
            if st.button("", icon=":material/refresh:", help="Reset"):
                reset_app()

        agents = ["Doctor", "Critic", "Supporter", "Summarizer"]
        icons = [" 🧑‍⚕️ ", " ✍️ ", " ✨ ", " 🧾 "]

        cols = st.columns(4)
        for i, (icon, name) in enumerate(zip(icons, agents)):
            with cols[i]:
                if st.button(
                    icon,
                    key=f"agent_{i}",
                    disabled=st.session_state.loading,
                    help=name,
                ):
                    st.session_state.selected_agent = name

        status_placeholder = st.empty()
        progress_placeholder = st.empty()

        if not st.session_state.loading and st.session_state.result is None:
            status_placeholder.caption("Agents are idle and ready.")
        elif st.session_state.loading:
            status_placeholder.caption("Agents are collaborating…")
            col1, col2 = st.columns([4, 1])
            with col1:
                progress_placeholder.progress(st.session_state.progress)
            with col2:
                st.markdown(f"**{st.session_state.progress}%**")
        else:
            status_placeholder.success("All agents completed successfully.")

        if st.session_state.selected_agent and not st.session_state.loading:
            show_agent_dialog(st.session_state.selected_agent)

# PROGRESS LOOP
if st.session_state.loading and st.session_state.result is None:
    st.session_state.progress = min(st.session_state.progress + 2, 98)

    if st.session_state.progress >= 98:
        st.session_state.result = run_pipeline(
            st.session_state.profile,
            st.session_state.goal,
        )
        st.session_state.agent_outputs = {
            item["agent"]: item for item in st.session_state.result.get("agents", [])
        }

        st.session_state.progress = 100
        st.session_state.loading = False

    st.rerun()

# RIGHT COLUMN — REPORT
with right:
    with st.container(key="gradient-main"):
        header_l, header_r = st.columns([4, 1])

        with header_l:
            st.subheader("📋 Your Personalized Plan")

        if st.session_state.result:
            raw_plan = st.session_state.result["plan_text"]["raw_plan"]
            structured = parse_plan_into_weeks(raw_plan)

            with header_r:
                pdf = generate_pdf(st.session_state.goal, structured)
                st.download_button(
                    "Download PDF",
                    pdf,
                    "health_report.pdf",
                    "application/pdf",
                )

            render_report(st.session_state.goal, structured)
        else:
            st.info("Enter your profile & goal to generate a plan")

# FOOTER
st.markdown(
    '<div class="footer">Powered by Multi-Agent AI • PubMed-backed Evidence</div>',
    unsafe_allow_html=True,
)
