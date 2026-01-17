# import streamlit as st
# import time
# from planner import generate_plan

# # -------------------------------
# # Page config
# # -------------------------------
# st.set_page_config(page_title="HealthAgents", layout="wide")

# # -------------------------------
# # Load CSS once
# # -------------------------------
# if "css_loaded" not in st.session_state:
#     with open("styles.css") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
#     st.session_state.css_loaded = True

# # -------------------------------
# # Session state defaults
# # -------------------------------
# defaults = {
#     "started": False,
#     "loading": False,
#     "plan": None,
#     "progress": 0,
#     "profile_text": "",
#     "goal_text": "",
# }
# for k, v in defaults.items():
#     st.session_state.setdefault(k, v)

# # -------------------------------
# # Layout
# # -------------------------------
# left, right = st.columns([1, 2.2], gap="large")

# # -------------------------------
# # Left: Input
# # -------------------------------
# with left:
#     with st.container(key="card-profile"):
#         st.subheader("Profile & Goal")

#         st.text_area(
#             "Profile",
#             placeholder="Age, lifestyle, preferences",
#             key="profile_text",
#             height=100,
#         )

#         st.text_input(
#             "Goal",
#             placeholder="e.g. improve knee strength",
#             key="goal_text",
#         )

#         if st.button("Generate My Plan"):
#             if not st.session_state.goal_text.strip():
#                 st.warning("Please enter a goal.")
#             else:
#                 st.session_state.started = True
#                 st.session_state.loading = True
#                 st.session_state.progress = 0
#                 st.session_state.plan = None

# # -------------------------------
# # Fake progress → real backend call
# # -------------------------------
# if st.session_state.loading:
#     st.session_state.progress += 2
#     time.sleep(0.03)

#     if st.session_state.progress >= 100:
#         with st.spinner("Generating plan…"):
#             st.session_state.plan = generate_plan(
#                 profile=st.session_state.profile_text,
#                 goal=st.session_state.goal_text,
#             )
#         st.session_state.loading = False

#     st.rerun()

# # -------------------------------
# # Right: Output
# # -------------------------------
# with right:
#     with st.container(key="gradient-main"):
#         st.subheader("Your Personalized Plan")

#         if st.session_state.plan:
#             plan = st.session_state.plan

#             st.markdown("### 🎯 Primary Plan")
#             st.json(plan.get("plan_primary", {}))

#             st.markdown("### 🛟 Safety")
#             st.json(plan.get("safety", {}))

#             if plan.get("plan_alternatives"):
#                 st.markdown("### 🔁 Alternatives")
#                 st.json(plan["plan_alternatives"])

#         else:
#             st.info("Generate a plan to see recommendations.")
import streamlit as st
import time
import json
from planner import generate_plan as generate_plan_local

# ===================================================
# PAGE CONFIG + CSS
# ===================================================
st.set_page_config(page_title="HealthAgents", layout="wide")

# Load CSS once
if "css_loaded" not in st.session_state:
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        # st.session_state.css_loaded = True    
    except FileNotFoundError:
        pass

# ===================================================
# SESSION STATE - BULLETPROOF
# ===================================================
if "loading" not in st.session_state:
    st.session_state.loading = False
if "plan" not in st.session_state:
    st.session_state.plan = None
if "error" not in st.session_state:
    st.session_state.error = None
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "profile_text" not in st.session_state:
    st.session_state.profile_text = ""
if "goal_text" not in st.session_state:
    st.session_state.goal_text = ""
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None

# ===================================================
# NAVBAR
# ===================================================
n1, n2, n3 = st.columns([2, 6, 2])
with n1:
    st.markdown("### 🩺 FIT4YOU")
with n2:
    st.markdown("""
        <div class="nav-right">
            <span class="nav-item">Overview</span>
            <span class="nav-item">How it works</span>
            <span class="nav-active">My Plan</span>
            <span class="nav-item">About</span>
        </div>
        """, unsafe_allow_html=True)
with n3:
    st.markdown("""
        <div class="user-info">
            <div>
                <div class="user-label">Logged in as</div>
                <div class="user-name">student</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ===================================================
# MAIN LAYOUT
# ===================================================
left, right = st.columns([1, 2.2], gap="large")

# ===================================================
# LEFT: INPUT + AGENTS
# ===================================================
with left:
    # Profile & Goal Input
    with st.container(key="card-profile"):
        st.subheader("👤 Profile & Goal")
        
        st.session_state.profile_text = st.text_area(
            "Profile",
            value=st.session_state.profile_text,
            placeholder="30yo female, 50kg, bad knees, desk job...",
            height=100,
            key="profile_input"
        )
        
        st.session_state.goal_text = st.text_input(
            "Goal",
            value=st.session_state.goal_text,
            placeholder="e.g. gain strength, lose weight",
            key="goal_input"
        )

        if st.button("🚀 Generate My Plan", type="primary", disabled=st.session_state.loading):
            if not st.session_state.goal_text.strip():
                st.warning("⚠️ Please enter a fitness goal.")
            else:
                st.session_state.loading = True
                st.session_state.plan = None
                st.session_state.error = None
                st.session_state.progress = 0
                st.session_state.selected_agent = None
                st.rerun()

    # Agents Working
    with st.container(key="card-agents"):
        col1, col2 = st.columns([8, 2])
        with col1:
            st.subheader("🤖 Agents Collaborating")
            st.caption("👆 Click agent icons to view contributions")
        
        with col2:
            if st.button("🔄 Reset", disabled=st.session_state.loading):
                st.session_state.loading = False
                st.session_state.plan = None
                st.session_state.error = None
                st.session_state.progress = 0
                st.session_state.selected_agent = None
                st.rerun()

        agents = [
            ("🧑‍⚕️", "Safety Agent"), 
            ("✍️", "Planner"), 
            ("🔄", "Alternatives"),
            ("🎯", "Coach")
        ]
        
        cols = st.columns(len(agents))
        for i, (icon, name) in enumerate(agents):
            with cols[i]:
                # ✅ FIXED: Always boolean
                agent_disabled = st.session_state.loading or bool(st.session_state.selected_agent)
                if st.button(icon, key=f"agent_{i}", help=name, disabled=agent_disabled):
                    st.session_state.selected_agent = name
                    st.rerun()

    # Status
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        if st.session_state.loading:
            st.caption("🤖 AI agents analyzing your profile...")
        elif st.session_state.plan:
            st.success("✅ All agents completed successfully!")
        else:
            st.caption("👆 Enter profile & goal above")
    
    with status_col2:
        if st.session_state.loading:
            st.progress(st.session_state.progress / 100)
            st.markdown(f"**{int(st.session_state.progress)}%**")

# ===================================================
# RIGHT: OUTPUT + PROGRESS
# ===================================================
with right:
    st.subheader("📋 Your Personalized Fitness Plan")
    
    # Agent Dialog - FIXED
    if st.session_state.selected_agent and st.session_state.plan:
        with st.dialog(f"🤖 {st.session_state.selected_agent}", width="700"):
            st.markdown(f"### {st.session_state.selected_agent} Contribution")
            plan = st.session_state.plan
            
            if st.session_state.selected_agent == "Safety Agent":
                st.json(plan.get("safety", {}))
            elif st.session_state.selected_agent == "Planner":
                st.json(plan.get("plan_primary", {}))
            elif st.session_state.selected_agent == "Alternatives":
                st.json(plan.get("plan_alternatives", {}))
            else:  # Coach
                st.json(plan)
            
            if st.button("✕ Close", key="close_dialog"):
                st.session_state.selected_agent = None
                st.rerun()

    # LOADING + API CALL
    # if st.session_state.loading:
    #     with st.spinner("🤖 Generating your personalized plan..."):
    #         # st.session_state.progress += 2
    #         # time.sleep(0.1)
            
    #         if st.session_state.progress >= 100:
    #             try:
    #                 st.session_state.plan = generate_plan_local(
    #                     profile=st.session_state.profile_text,
    #                     goal=st.session_state.goal_text
    #                 )
    #                 st.session_state.loading = False
    #             except Exception as e:
    #                 st.session_state.error = str(e)
    #                 st.session_state.loading = False
    #             st.rerun()
    if st.session_state.loading:
        with st.spinner("Generating plan..."):
            try:
                st.session_state.plan = generate_plan_local(
                    profile=st.session_state.profile_text,
                    goal=st.session_state.goal_text
                )
                st.session_state.loading = False
            except Exception as e:
                st.session_state.error = str(e)
                st.session_state.loading = False
        st.rerun()

    # ERROR STATE
    elif st.session_state.error:
        st.error(f"❌ **Error:** {st.session_state.error}")
        if "Backend error" in st.session_state.error:
            st.info("💡 **Backend:** `cd backend && python app.py`\n💡 **Ollama:** `ollama list`")
        
        if st.button("🔄 Try Again", type="secondary"):
            st.session_state.loading = False
            st.session_state.error = None
            st.session_state.plan = None
            st.session_state.progress = 0
            st.session_state.selected_agent = None
            st.rerun()

    # SUCCESS STATE - BEAUTIFUL LAYOUT
    elif st.session_state.plan:
        st.success("✅ Your personalized fitness plan is ready!")
        plan = st.session_state.plan
        
        # ROW 1: Goal + Week Preview + Safety
        r1c1, r1c2, r1c3 = st.columns(3, gap="large")
        
        with r1c1:
            st.markdown("### 🎯 **Goal**")
            goal = plan.get('metadata', {}).get('goal', st.session_state.goal_text)
            st.markdown(f"**{goal[:50]}**")
            st.caption("Personalized for your profile")
        
        with r1c2:
            st.markdown("### 📅 **Week 1 Preview**")
            week1 = plan.get("plan_primary", {}).get("week_1", {})
            for day, exercises in list(week1.items())[:3]:
                display_ex = str(exercises)[:40] + "..." if len(str(exercises)) > 40 else str(exercises)
                st.markdown(f"**{day.upper()}**: {display_ex}")
        
        with r1c3:
            st.markdown("### 🛡️ **Safety First**")
            risks = plan.get("safety", {}).get("risks", [])
            if risks:
                st.warning(risks[0][:120])
            else:
                st.success("✅ Safe for your profile")

        st.markdown("---")
        
        # Primary Plan
        with st.expander("🎯 **Primary Training Plan** (Click agents above)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📅 Week 1**")
                week1_data = {k: v for k, v in plan.get("plan_primary", {}).get("week_1", {}).items()}
                st.json(week1_data)
            
            with col2:
                st.markdown("**📅 Week 2**")
                week2_data = {k: v for k, v in plan.get("plan_primary", {}).get("week_2", {}).items()}
                st.json(week2_data)

        # Alternatives
        alternatives = plan.get("plan_alternatives")

        # Normalize to list
        if not isinstance(alternatives, list):
            alternatives = []

        if alternatives:
            st.markdown("### 🔄 **Alternative Plans**")

            # Limit to max 2 cards
            visible_alts = alternatives[:2]
            alt_cols = st.columns(len(visible_alts))

            for i, alt in enumerate(visible_alts):
                # Defensive dict access
                if not isinstance(alt, dict):
                    continue

                with alt_cols[i]:
                    st.markdown(
                        f"**{alt.get('plan_name', f'Plan {i+1}')[:35]}**"
                    )
                    description = alt.get("description", "")
                    if description:
                        st.caption(description[:100])
                    else:
                        st.caption("Alternative training option")
        else:
            st.info("No alternative plans available at this time.")

        # Additional Info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            if plan.get("plan_primary", {}).get("nutrition_timing"):
                st.markdown("### 🍎 **Nutrition**")
                st.caption(plan.get("plan_primary", {}).get("nutrition_timing", "")[:150])
        
        with col_info2:
            if plan.get("plan_primary", {}).get("habits"):
                st.markdown("### 💡 **Daily Habits**")
                habits = plan.get("plan_primary", {}).get("habits", [])
                for habit in habits[:3]:
                    st.caption(f"• {habit}")

        # Questions
        if plan.get("questions_to_user"):
            st.markdown("### ❓ **Next Steps**")
            for q in plan["questions_to_user"][:4]:
                st.write(f"• {q}")

        # Download
        st.markdown("---")
        st.download_button(
            "📥 Download Full Plan",
            data=json.dumps(plan, indent=2, ensure_ascii=False),
            file_name=f"fit4you_plan_{int(time.time())}.json",
            mime="application/json"
        )

    else:
        st.info("👆 **Enter your profile & goal** → **Generate My Plan**")

# Footer
st.markdown("---")
st.markdown("*Powered by Ollama + Multi-Agent AI 🚀*")
