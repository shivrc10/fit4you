# app.py - COMPLETE WORKING VERSION
import streamlit as st
import time
from planner import generate_plan

# ===================================================
# AGENT DIALOG
# ===================================================
@st.dialog("Agent details")
def show_agent_dialog(agent_name: str):
    st.markdown(f"### {agent_name}")
    st.markdown(
        """
        **Status:**  
        Details about this agent's contribution to your plan.
        """
    )
    if st.button("Close"):
        st.session_state.selected_agent = None
        st.rerun()

# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(page_title="HealthAgents", layout="wide")

# ===================================================
# LOAD CSS
# ===================================================
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===================================================
# SESSION STATE INITIALIZATION
# ===================================================
defaults = {
    "started": False,
    "loading": False,
    "plan": None,
    "progress": 0,
    "agent_step": 0,
    "completed_agents": set(),
    "profile_text": "",
    "goal_text": "",
    "selected_agent": None,
}

for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ===================================================
# RESET FUNCTION
# ===================================================
def reset_app():
    for key in ["profile_text", "goal_text"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.started = False
    st.session_state.loading = False
    st.session_state.plan = None
    st.session_state.progress = 0
    st.session_state.agent_step = 0
    st.session_state.completed_agents = set()
    st.session_state.selected_agent = None

# ===================================================
# NAVBAR
# ===================================================
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

# ===================================================
# MAIN LAYOUT
# ===================================================
left, right = st.columns([1, 2.2], gap="large")

# ===================================================
# LEFT COLUMN — INPUT + AGENTS
# ===================================================
with left:
    # -------------------------------
    # PROFILE INPUT
    # -------------------------------
    with st.container(key="card-profile"):
        st.subheader("Profile & Goal Input")

        profile = st.text_area(
            "Profile",
            placeholder="Age, lifestyle, preferences (free text)",
            height=90,
            key="profile_text",
        )

        goal = st.text_input(
            "Goal",
            placeholder="e.g. weight loss, fitness",
            key="goal_text",
        )

        if st.button("Generate My Plan"):
            if not goal.strip():
                st.warning("Please enter a goal.")
            else:
                st.session_state.started = True
                st.session_state.loading = True
                st.session_state.progress = 0
                st.session_state.agent_step = 0
                st.session_state.completed_agents = set()
                st.session_state.plan = None

    # -------------------------------
    # AGENTS WORKING
    # -------------------------------
    with st.container(key="card-agents"):
        hcol, bcol = st.columns([8, 2])

        with hcol:
            st.subheader("Agents Working")
            st.caption("ℹ️ Click agent icon to view its contribution")

        with bcol:
            if st.button("", icon=":material/refresh:", help="Regenerate plan"):
                reset_app()
                st.rerun()

        agents = [
            ("🧑‍⚕️", "Doctor"),
            ("✍️", "Critic"),
            ("✨", "Supporter"),
            ("🎯", "Coach"),
        ]

        cols = st.columns(len(agents))

        for i, (icon, name) in enumerate(agents):
            with cols[i]:
                # Determine button state
                disabled = st.session_state.loading or st.session_state.selected_agent is not None
                
                # Agent button - NATIVE st.button styled by CSS
                clicked = st.button(
                    icon,
                    key=f"agent_btn_{i}",
                    help=name,
                    disabled=disabled
                )
                
                # Handle click immediately
                if clicked and not st.session_state.loading:
                    st.session_state.selected_agent = name

        # -------------------------------
        # STATUS + PROGRESS
        # -------------------------------
        status_placeholder = st.empty()
        hint_placeholder = st.empty()
        progress_placeholder = st.empty()

        if not st.session_state.started:
            status_placeholder.caption("Agents are idle and ready.")
            progress_placeholder.empty()

        elif st.session_state.loading:
            status_placeholder.caption("Agents are collaborating…")
            # progress_placeholder.progress(st.session_state.progress)
            col1, col2 = st.columns([4, 1])
            with col1:
                progress_placeholder.progress(st.session_state.progress)
            with col2:
                st.markdown(f"**{st.session_state.progress}%**")

        else:
            progress_placeholder.empty()
            status_placeholder.success("All agents completed successfully.")

        # ===================================================
        # DIALOG TRIGGER - SINGLE CALL AFTER LOOP
        # ===================================================
        if st.session_state.selected_agent and not st.session_state.loading:
            show_agent_dialog(st.session_state.selected_agent)

# ===================================================
# GENERATION LOGIC
# ===================================================
if st.session_state.loading and st.session_state.selected_agent is None:
    st.session_state.progress += 1
    time.sleep(0.04)

    p = st.session_state.progress

    if p < 25:
        st.session_state.agent_step = 0
    elif p < 50:
        st.session_state.agent_step = 1
    elif p < 75:
        st.session_state.agent_step = 2
    else:
        st.session_state.agent_step = 3

    if p == 25:
        st.session_state.completed_agents.add(0)
    if p == 50:
        st.session_state.completed_agents.add(1)
    if p == 75:
        st.session_state.completed_agents.add(2)

    if p >= 100:
        st.session_state.completed_agents.add(3)
        st.session_state.plan = generate_plan(
            {"profile": profile, "goal": goal}
        )
        st.session_state.loading = False

    st.rerun()

# ===================================================
# RIGHT COLUMN — OUTPUT
# ===================================================
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
            st.info("Generate a plan to see recommendations.")

# ===================================================
# FOOTER
# ===================================================
st.markdown(
    '<div class="footer">HealthAgents · Research Prototype</div>',
    unsafe_allow_html=True,
)
