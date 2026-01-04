import streamlit as st
import time
from planner import generate_plan

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="HealthAgents", layout="wide")

# ---------------------------------------------------
# LOAD CSS
# ---------------------------------------------------
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
st.session_state.setdefault("loading", False)
st.session_state.setdefault("plan", None)
st.session_state.setdefault("progress", 0)

# ---------------------------------------------------
# NAVBAR
# ---------------------------------------------------
n1, n2, n3 = st.columns([2, 6, 2])

with n1:
    st.markdown("### 🩺 HealthAgents")

with n2:
    st.markdown(
        """
        <div class="nav-right">
            <span class="nav-item">Overview</span>
            <span class="nav-item">How it works</span>
            <span class="nav-active">My Plan</span>
            <span class="nav-item">About</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with n3:
    st.markdown(
        """
        <div class="user-info">
            <div>
                <div class="user-label">Logged in as</div>
                <div class="user-name">tanvi</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------
left, right = st.columns([1, 2.2], gap="large")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------
with left:
    with st.container(key="card-profile"):
        st.subheader("Profile & Goal")
        st.caption("Enter information to generate your plan")

        profile = st.text_area(
            "Profile",
            placeholder="Age, lifestyle, preferences (free text)",
            height=100,
        )

        goal = st.text_input(
            "Goal",
            placeholder="e.g. lose weight, improve stamina",
        )

        error = st.empty()

        if st.button("Generate My Plan"):
            if not goal.strip():
                error.warning("Please enter a goal.")
            else:
                error.empty()
                st.session_state.loading = True
                st.session_state.plan = None
                st.session_state.progress = 0

# ---------------------------------------------------
# AGENTS WORKING — INLINE (ONLY WHEN LOADING)
# ---------------------------------------------------
if st.session_state.loading:
    st.markdown("### Generating your plan")

    status = st.empty()
    bar = st.progress(0)
    agent_ui = st.empty()

    steps = [
        ("Analyzing profile", 30, 0),
        ("Generating recommendations", 65, 1),
        ("Finalizing plan", 100, 2),
    ]

    icons = ["🧠", "📋", "✅"]

    for text, target, idx in steps:
        status.info(text)
        for i in range(st.session_state.progress, target):
            time.sleep(0.02)
            st.session_state.progress = i
            bar.progress(i)

            row = '<div class="agent-row">'
            for j, icon in enumerate(icons):
                if j < idx:
                    cls = "agent-icon completed"
                elif j == idx:
                    cls = "agent-icon active"
                else:
                    cls = "agent-icon"
                row += f'<div class="{cls}">{icon}</div>'
            row += "</div>"

            agent_ui.markdown(row, unsafe_allow_html=True)

    st.session_state.plan = generate_plan(
        {"profile": profile, "goal": goal}
    )
    st.session_state.loading = False

# ---------------------------------------------------
# OUTPUT SECTION
# ---------------------------------------------------
with right:
    with st.container(key="gradient-main"):
        st.subheader("Your Personalized Plan")

        if st.session_state.plan:
            c1, c2, c3 = st.columns(3)

            with c1:
                with st.container(key="plan-morning"):
                    st.markdown("### ☀ Morning")
                    st.markdown("- Balanced meals\n- Light activity")

            with c2:
                with st.container(key="plan-afternoon"):
                    st.markdown("### 🕒 Afternoon")
                    st.markdown("- Movement breaks\n- Hydration")

            with c3:
                with st.container(key="plan-evening"):
                    st.markdown("### 🌙 Evening")
                    st.markdown("- Light dinner\n- Wind-down routine")

            st.download_button(
                "📄 Download Plan",
                data=str(st.session_state.plan),
                file_name="health_plan.txt",
            )
        else:
            st.info("Generate a plan to see recommendations")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown(
    '<div class="footer">HealthAgents · Research Prototype</div>',
    unsafe_allow_html=True,
)
