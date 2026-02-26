import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.preprocessor import preprocess, detect_peaks
from utils.llm import interpret_spectrum
from utils.beer_lambert import calculate_concentration

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

tab1, tab2 = st.tabs(["Spectrum Interpretation", "Beer-Lambert Analysis"])

with tab1:
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

with tab2:
    st.subheader("Beer-Lambert Concentration")
    st.markdown("Calculate sample concentration using **A = εlc**")

    st.divider()

    col_x, col_y = st.columns(2)

    with col_x:
        absorbance_input = st.number_input(
            "Absorbance (A)",
            min_value=0.0,
            max_value=10.0,
            value=0.85,
            step=0.01,
            help="The absorbance value at λmax"
        )
        epsilon_input = st.number_input(
            "Molar Absorptivity ε (L·mol⁻¹·cm⁻¹)",
            min_value=1.0,
            value=13000.0,
            step=100.0,
            help="The molar absorptivity constant for your compound at λmax"
        )
        path_length_input = st.number_input(
            "Path Length l (cm)",
            min_value=0.1,
            value=1.0,
            step=0.1,
            help="The path length of your cuvette (standard = 1.0 cm)"
        )
    
    with col_y:
        st.markdown("***Formula***")
        st.latex(r"A = \varepsilon \cdot l \cdot c")
        st.latex(r"c = \frac{A}{\varepsilon \cdot l}")
        st.markdown("Where:")
        st.markdown("- **A** = Absorbance (unitless)")
        st.markdown("- **ε** = Molar absorptivity (L·mol⁻¹·cm⁻¹)")
        st.markdown("- **l** = Path length (cm)")
        st.markdown("- **c** = Concentration (mol/L)")
    
    st.divider()

    if st.button("Calculate Concentration", type="primary"):
        try:
            result = calculate_concentration(
                absorbance=absorbance_input,
                epsilon=epsilon_input,
                path_length=path_length_input
            )
            col_r1, colr2, = st.columns(2)
            col_r1.metric(
                "Concentration",
                f"{result['concentration']:.6f} mol/L",
                f"{result['concentration'] * 1000:.4f} mmol/L"
            )
            col_r2.markdown(f"**Measurement Reliability:**\n\n{result['reliability']}")
        except ValueError as e:
            st.error(f"Input Error: {e}")