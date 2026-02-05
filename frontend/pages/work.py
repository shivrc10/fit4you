import streamlit as st

# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(page_title="How it works – FIT4YOU", layout="wide")


# ===================================================
# LOAD CSS (same behavior as app.py)
# ===================================================
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass



logo_col, spacer, nav1, nav2, nav3 = st.columns(
    [2.2, 6.0, 1.2, 1.2, 1.2]
)

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
# CONTENT — CARD LAYOUT (CASUAL / STARTUP STYLE)
# ===================================================
st.markdown("## How FIT4YOU Works")
# st.caption("Fitness advice that thinks first, plans second")
# st.divider()

st.markdown(
    """
<div class="st-key-gradient-main">

  <div class="st-key-card-profile">
    <h3>🎯 Start with your goal</h3>
    <p title="Your input is used for personalization, not medical facts.">
      Share your goal and what matters to you — schedule, preferences, or physical limits.
      This helps tailor the plan to your life.
    </p>
  </div>

  <div class="st-key-card-profile">
    <h3>📊 We check the science</h3>
    <p title="All medical reasoning is based on real, peer-reviewed studies.">
      Before giving advice, FIT4YOU checks real medical research from trusted sources.
      No trends. No internet myths.
    </p>
  </div>

  <div class="st-key-card-profile">
    <h3>⚖️ Multiple perspectives review it</h3>
    <p title="Multiple AI agents review the same question from different angles.">
      Instead of one AI opinion, a small AI team reviews your request and 
      check benefits, risks, and fairness.
    </p>
  </div>
  
  <div class="st-key-card-profile">
    <h3>🧾 You approve every step</h3>
    <p title="Nothing moves forward without your approval.">
        You see a clear summary of pros, risks, and evidence strength.
        The plan continues only if you approve.
      </p>
  </div>


  <div class="st-key-card-profile">
    <h3>🏋️ Your plan, tailored to you</h3>
    <p title="Plans are practical, safety-aware, and easy to follow.">
      After approval, FIT4YOU creates a simple, structured fitness plan
      adapted to your needs.
    </p>
  </div>

  <div class="st-key-card-profile" style="border-left: 4px solid #10b981;">
    <h3>🛡️ Safety by design</h3>
    <p title="The system favors care and transparency over speed.">
      Built-in checks, clear explanations, and quality controls
      help keep recommendations safe and trustworthy.
    </p>
  </div>

</div>
""",
    unsafe_allow_html=True
)


# ===================================================
# FOOTER
# ===================================================
st.markdown(
    '<div class="footer">Powered by Multi-Agent AI • PubMed-backed Evidence • Human-Centered Design</div>',
    unsafe_allow_html=True,
)
