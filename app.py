import streamlit as st

st.set_page_config(
    page_title="UV-VIS Spectralyzer",
    page_icon="/Users/larsius/Desktop/codex/UV-Vis-LLM/spectralyzer.png",
    layout="wide"
)

st.markdown("""
        <style>
        [data-testid="stImage"] {
            display: flex;
            align-items: center;
        }
        </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 14])

with col1:
    st.image("/Users/larsius/Desktop/codex/UV-Vis-LLM/spectralyzer.png", width=100)

with col2:
    st.markdown("<h1 style='margin-top: 5px;'>Spectralyzer</h1", unsafe_allow_html=True)
    st.caption("Upload a spectrum, get an interpretation powered by AI.")

st.info("App is under construction. Phase 1 complete!")