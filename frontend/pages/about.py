import streamlit as st

# ===================================================
# PAGE CONFIG
# ===================================================
st.set_page_config(page_title="About – FIT4YOU", layout="wide")


# ===================================================
# LOAD CSS (same behavior as app.py)
# ===================================================
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass


logo_col, spacer, nav1, nav2, nav3 = st.columns([2.2, 6.0, 1.2, 1.2, 1.2])

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
# CONTENT
# ===================================================
st.markdown("## About FIT4YOU")
# st.caption("Research-grade • Transparent • Safety-oriented")
# st.divider()

st.markdown(
    """
<div class="st-key-gradient-main">

  <!-- CARD 1: WHAT IT IS -->
  <div class="st-key-card-profile">
    <h3>What FIT4YOU is</h3>
    <p>
      <strong>FIT4YOU (HealthAgents)</strong> is a research-grade multi-agent AI system
      designed to explore how health and fitness recommendations can be grounded in
      <strong>real scientific evidence</strong>.
    </p>
  </div>

  <!-- CARD 2: WHAT MAKES IT DIFFERENT -->
  <div class="st-key-card-profile">
    <h3>What makes it different</h3>
    <ul>
      <li>Uses peer-reviewed PubMed literature</li>
      <li>Separates reasoning into explicit AI roles</li>
      <li>Makes decision paths visible and reviewable</li>
    </ul>
  </div>

  <!-- CARD 3: DESIGN PHILOSOPHY -->
  <div class="st-key-card-profile">
    <h3>Design philosophy</h3>
    <ul>
      <li>Evidence over intuition</li>
      <li>Transparency over black boxes</li>
      <li>Human control by design</li>
    </ul>
  </div>
  
  <div class="st-key-card-profile">
    <h3>Role of the human</h3>
    <ul>
      <li>FIT4YOU does not replace human judgment.</li>
      <li>Instead, it supports users by surfacing evidence, risks,
      and reasoning so decisions can be made deliberately.</li>
    </ul>
  </div>

  <!-- CARD 4: SCOPE & CONTEXT -->
  <div class="st-key-card-profile" style="border-left: 4px solid #f97316;">
    <h3>Scope and context</h3>
    <p>
      FIT4YOU does <strong>not</strong> provide medical diagnosis or treatment.
      It is intended for <strong>research and decision support only</strong>.
    </p>
    <p style="margin-bottom: 0;">
      Developed as part of an <strong>HCAI / AI Safety research project</strong>.
    </p>
  </div>

</div>
""",
    unsafe_allow_html=True,
)


# ===================================================
# FOOTER
# ===================================================
st.markdown(
    '<div class="footer">Powered by Multi-Agent AI • PubMed-backed Evidence</div>',
    unsafe_allow_html=True,
)
