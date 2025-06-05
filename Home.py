import streamlit as st
from datetime import date

st.set_page_config(page_title="Welcome to IFRS13 Classification model", layout="wide")
st.sidebar.markdown("""
### IFRS13 Classification Workflow

Navigate through:
- **Model Inference**
- **Risk Factor Testing**
- **Rationale Explanation**

Use the sidebar to choose a module.
""")

st.markdown("""
<style>
.big-title {
    font-size: 2.4em;
    font-weight: bold;
    color: #003366;
    margin: 0.67em 0;
    font-family: 'Segoe UI', sans-serif;
}
.subtitle {
    font-size: 1.em;
    color: #333;
    margin-bottom: 1.5em;
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="big-title"> Welcome to the IFRS13 Fair Value Classification Application</div>
<div class="subtitle">A modular, explainable, and friendly classification tool - to simplify and streamline the process of fair value classification</div>
""", unsafe_allow_html=True)

st.info("Use the navigation on the left sidebar to begin.")

st.markdown("""
---
### 📌 Key Features:

- **Modular Design:** Split workflows for Model Inference, Observability, and Rationale
- **Regulatory Focus:** Built for IFRS13 compliance scenarios
- **Explainable AI:** Combines ML models and GPT rationale
- **Azure-native:** Integrates with Azure ML & OpenAI services

---
📅 Today's Date: **{}**
""".format(date.today().strftime("%d %b %Y")))
