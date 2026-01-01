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
for k, v in {"loading": False, "plan": None, "progress": 0, "cancel": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
            <span class="nav-item">Home</span>
            <span class="nav-item">Profile</span>
            <span class="nav-active">My Plan</span>
            <span class="nav-item">About</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Temporary mock user (replace later with real auth)
username = "tanvi"

with n3:
    st.markdown(
        f"""
        <div class="user-info">
            <span class="user-icon">👤</span>
            <div class="user-text">
                <span class="user-label">Logged in as</span>
                <span class="user-name">{username}</span>
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
# LEFT COLUMN
# ---------------------------------------------------
with left:
    # Profile card
    with st.container(key="card-profile"):
        st.subheader("Profile & Goal Input")
        st.caption("Tell us a bit about yourself to generate a plan")

        age = st.text_input("Profile")
        goal = st.text_input("Goal", placeholder="e.g. lose weight, improve stamina")

        error_box = st.empty()

        if st.button("Generate My Plan"):
            if not goal.strip():
                error_box.warning("Please describe your goal so we can personalize your plan.")
            else:
                error_box.empty()
                st.session_state.loading = True
                st.session_state.plan = None
                st.session_state.progress = 0
                st.session_state.cancel = False

    # Agents card
    # with st.container(key="card-agents"):
    #     st.subheader("Agents Working")

    #     bar = st.progress(st.session_state.progress)
    #     status = st.empty()

    #     cancel_placeholder = st.empty()

    #     if st.session_state.loading:
    #         with cancel_placeholder.container():
    #             if st.button(
    #                 "Cancel generation", key="cancel_gen", help="Stop plan generation"
    #             ):
    #                 st.session_state.cancel = True
    #                 st.session_state.loading = False
    #                 status.warning("Generation cancelled")
    #                 st.stop()

    #         steps = [
    #             ("🍎 Nutrition agent analyzing", 30),
    #             ("🏃 Exercise agent planning", 60),
    #             ("😴 Sleep agent optimizing", 90),
    #             ("📊 Finalizing plan", 100),
    #         ]

    #         for text, target in steps:
    #             if st.session_state.cancel:
    #                 break
    #             status.info(text)
    #             for i in range(st.session_state.progress, target, 4):
    #                 time.sleep(0.05)
    #                 st.session_state.progress = i
    #                 bar.progress(i)

    #         if not st.session_state.cancel:
    #             st.session_state.plan = generate_plan({"age": age, "goal": goal})
    #             st.session_state.loading = False
    #             status.success("✅ All agents completed successfully")

    #     else:
    #         status.caption("Waiting to generate your personalized plan")
    
    with st.container(key="card-agents"):
        st.subheader("Agents Working")
        status = st.empty()
        bar = st.progress(st.session_state.progress)

        if not st.session_state.loading and not st.session_state.plan:
            status.caption("Agents are waiting for your input.")

        elif st.session_state.loading:
            focus = infer_focus(goal)

            steps = [
                ("Understanding your goal…", 15),
                ("🍎 Nutrition agent working…" if focus in ["nutrition", "balanced"] else "🍎 Nutrition agent waiting…", 40),
                ("🏃 Exercise agent working…" if focus in ["exercise", "balanced"] else "🏃 Exercise agent waiting…", 70),
                ("😴 Sleep agent optimizing recovery…", 90),
                ("Finalizing your plan…", 100)
            ]

            for text, target in steps:
                if st.session_state.cancel:
                    status.warning("Generation cancelled.")
                    break

                status.info(text)
                for i in range(st.session_state.progress, target):
                    time.sleep(0.03)
                    st.session_state.progress = i
                    bar.progress(i)

            if not st.session_state.cancel:
                st.session_state.plan = generate_plan({"age": age, "goal": goal})
                st.session_state.loading = False
                status.success("All agents completed successfully.")

# ---------------------------------------------------
# RIGHT COLUMN — PLAN OUTPUT (NO SCORE)
# ---------------------------------------------------
with right:
    with st.container(key="gradient-main"):
        st.subheader("Your Personalized Plan")

        if st.session_state.plan:
            plan = st.session_state.plan

            c1, c2, c3 = st.columns(3)

            with c1:
                with st.container(key="plan-morning"):
                    st.markdown("### ☀ Morning Routine")
                    st.markdown("- Balanced nutrition\n- Regular meal timing")

            with c2:
                with st.container(key="plan-afternoon"):
                    st.markdown("### 🕒 Afternoon Focus")
                    st.markdown("- 20–30 min moderate activity\n- Stretch breaks")

            with c3:
                with st.container(key="plan-evening"):
                    st.markdown("### 🌙 Evening Wind-down")
                    st.markdown("- Light dinner\n- Screen-free before bed")

            st.download_button(
                "📄 Download Plan", data=str(plan), file_name="health_plan.txt"
            )

        else:
            st.info("Your plan will appear here after generation")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown(
    '<div class="footer">HealthAgents · v0.1 · Research Prototype</div>',
    unsafe_allow_html=True,
)
