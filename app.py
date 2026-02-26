import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.preprocessor import preprocess, detect_peaks
from utils.llm import interpret_spectrum

st.set_page_config(
    page_title="UV-VIS Spectralyzer",
    page_icon="spectralyzer.png",
    layout="wide"
)

st.markdown("""
        <style>
        [data-testid="stImage"] {
            display: flex;
            align-items: center;
        }
        a[data-testid="stHeaderActionElements"] {
            display: none;
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

# File Uploader
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
        # Preprocess
        wl, ab = preprocess(df)

        # Detect peaks
        peaks = detect_peaks(wl, ab)

        # Metrics
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Wavelength Range", f"{wl.min():.0f}–{wl.max():.0f} nm")
        col_b.metric("λmax", f"{wl[ab.argmax()]:.1f} nm")
        col_c.metric("Peaks Detected", len(peaks))

        st.divider()

        # Plot the spectrum
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=wl,
            y=ab,
            mode="lines",
            name="Absorbance",
            line=dict(color="#7B2FBE", width=2.5),
        ))

        # Mark peaks on chart
        for peak in peaks:
            color = "#FF4B4B" if peak["type"] == "major" else "#FFA500"
            fig.add_annotation(
                x=peak["wavelength_nm"],
                y=peak["absorbance"],
                text=f"{peak['wavelength_nm']} nm",
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                font=dict(color=color, size=11),
                ay=-40
            )

        fig.update_layout(
            title="UV-Vis Absorption Spectrum",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Absorbance",
            template="plotly_dark",
            hovermode="x unified",
            height=450
        )

        st.plotly_chart(fig, width='stretch')

        # Peaks table
        st.subheader("Detected Peaks")
        if peaks:
            peaks_df = pd.DataFrame(peaks)
            peaks_df.columns = ["Wavelength (nm)", "Absorbance", "Type"]
            st.dataframe(peaks_df, width='stretch')
        else:
            st.info("No significant peaks detected.")

st.divider()
st.subheader("Spectronalysis")

if st.button("Interpret Spectrum", type="primary"):
    with st.spinner("Analyzing spectrum with Spectrolyzer..."):
        try:
            interpretation = interpret_spectrum(
                peaks=peaks,
                wl_range=(float(wl.min()), float(wl.max())),
                lambda_max=float(wl[ab.argmax()])
            )
            st.markdown(interpretation)
        except Exception as e:
            st.error(f"LLM Error: {e}. Make sure Spectrolyzer is running with 'ollama serve'.")

else:
    st.info("Upload a CSV file to get started.")