import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

st.divider()

#File Uploader
uploaded_file = st.file_uploader(
    "Upload your spectrum (.csv)",
    type=["csv"],
    help="CSV must have two columns: wavelength_nm and absorbance"
)

if uploaded_file is not None:
    #Load the data
    df = pd.read_csv(uploaded_file)

    #Validate columns
    if "wavelength_nm" not in df.columns or "absorbance" not in df.columns:
        st.error("ERROR: Your CSV must have columns name 'wavelength_nm' and 'absorbance'.")
    else:
        #Show basic info
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Wavelength Range", f"{df['wavelength_nm'].min()}-{df['wavelength_nm'].max()} nm")
        col_b.metric("λmax", f"{df.loc[df['absorbance'].idxmax(), 'wavelength_nm']} nm")
        col_c.metric("Max Absorbance", f"{df['absorbance'].max():.4f}")

        st.divider()

        # Plot the spectrum
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["wavelength_nm"],
            y=df["absorbance"],
            mode="lines",
            name="Absorbance",
            line=dict(color="#7B2FBE", width=2.5),
        ))

        fig.update_layout(
            title="UV-Vis Absorption Spectrum",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Absorbance",
            template="plotly_dark",
            hovermode="x unified",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a CSV file to get started.")