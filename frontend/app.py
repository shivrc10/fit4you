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
# PAGE CONFIG
# ===================================================
st.set_page_config(
    page_title="FIT4YOU – My Plan",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================================================
# LOAD CSS
# ===================================================
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ===================================================
# NAVBAR
# ===================================================
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
    "result": None,
    "profile": "",
    "goal": "",
    "selected_agent": None,
    "agent_outputs": {},
    "pipeline_action": None, 
}

for k, v in DEFAULT_STATE.items():
    st.session_state.setdefault(k, v)

def reset_app():
    st.session_state.clear()
    st.rerun()



# ===================================================
# AGENT DIALOG
# ===================================================
@st.dialog("Agent details", width="medium")
def show_agent_dialog(agent_name: str):
    explanations = {
        "Doctor": "Reviews real PubMed evidence and identifies medical risks.",
        "Critic": "Challenges weak evidence and highlights limitations.",
        "Supporter": "Balances critique with realistic encouragement.",
        "Summary": "Condenses the multi-agent debate into a neutral overview.",
    }

    st.markdown(f"## {agent_name}")
    st.caption(explanations.get(agent_name, ""))

    agent_data = st.session_state.agent_outputs.get(agent_name)

    # st.divider()

    if agent_data:
        st.markdown(agent_data.get("content", "_No content available._"))

        if agent_name == "Doctor" and agent_data.get("sources"):
            st.divider()
            st.markdown("### Evidence sources")
            for pmid in agent_data["sources"]:
                st.markdown(f"- `{pmid}`")
    else:
        st.warning("No output available for this agent.")

    if st.button("Close", use_container_width=True):
        st.session_state.selected_agent = None
        st.rerun()

# ===================================================
# PDF
# ===================================================
def generate_pdf(goal: str, structured_plan: dict) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    x, y = 50, height - 50

    c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, "FIT4YOU Health Report")
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
# LAYOUT
# ===================================================
left, right = st.columns([1, 2.2], gap="large")

# LEFT
with left:
    with st.container(key="card-profile"):

        st.subheader("👤 Profile & Goal")

        st.text_area("Profile", key="profile", height=120,placeholder = "42yo male, 88kg, bad knees, desk job")
        st.text_input("Goal", key="goal", placeholder = "Lose fat and get strong without joint pain")

        if st.button("Generate Plan", type="primary"):
            st.session_state.loading = True
            st.session_state.pipeline_action = "evidence"
            st.session_state.result = None
            st.session_state.agent_outputs = {}
            st.session_state.selected_agent = None
            st.rerun()

    with st.container(key="card-agents"):
        h_l, h_r = st.columns([6, 1])

        with h_l:
            st.subheader("Agents Working")
            st.caption("ℹ️ Click agent icon to view its contribution")
        with h_r:
            if st.button("", icon=":material/refresh:", help="Clear Input"):
                reset_app()
        agents = ["Doctor", "Critic", "Supporter"]
        icons = [" 🧑‍⚕️ ", " ✍️ ", " ✨ "]

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

        if st.session_state.selected_agent and not st.session_state.loading:
            show_agent_dialog(st.session_state.selected_agent)

# ===================================================
# PIPELINE EXECUTION (ONLY PLACE IT RUNS)
# ===================================================
if st.session_state.loading and st.session_state.result is None:
    with st.spinner("Agents are collaborating…"):
        action = st.session_state.pipeline_action

        if action in ("evidence", "regenerate"):
            result = run_pipeline(
                st.session_state.profile,
                st.session_state.goal,
            )

        elif action == "proceed":
            result = run_pipeline(
                st.session_state.profile,
                st.session_state.goal,
                proceed=True,
            )

        else:
            result = None

        if result:
            st.session_state.result = result
            st.session_state.agent_outputs = {
                item["agent"]: item
                for item in result.get("agents", [])
            }

    st.session_state.loading = False
    st.session_state.pipeline_action = None
    st.rerun()

# ===================================================
# RIGHT
# ===================================================
with right:
    with st.container(key="gradient-main"):
        header_l, header_r = st.columns([4, 1])
        with header_l:
            st.subheader("📋 Your Personalized Plan")

        result = st.session_state.result

        if result is None:
            st.info("Enter your profile and goal to Generate Plan")

        elif result.get("awaiting_confirmation"):
            summary = st.session_state.agent_outputs.get("Summary")

            if summary:
                st.success("Evidence reviewed. Summary ready.")
                st.markdown("### Neutral Summary")
                st.markdown(summary["content"])

            col_yes, col_no = st.columns(2)

            with col_yes:
                if st.button("Agree & Generate Plan", use_container_width=True):
                    st.session_state.loading = True
                    st.session_state.pipeline_action = "proceed"
                    st.session_state.result = None
                    st.rerun()

            with col_no:
                if st.button("Regenerate Evidence", use_container_width=True):
                    st.session_state.loading = True
                    st.session_state.pipeline_action = "regenerate"
                    st.session_state.result = None
                    st.session_state.agent_outputs = {}
                    st.rerun()

        elif result.get("can_proceed"):
            raw_plan = result["plan_text"]["raw_plan"]
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
            st.warning("Unexpected pipeline state. Please regenerate.")

# ===================================================
# FOOTER
# ===================================================
st.markdown(
    '<div class="footer">Powered by Multi-Agent AI • PubMed-backed Evidence</div>',
    unsafe_allow_html=True,
)
