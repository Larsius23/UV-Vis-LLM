import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

def preprocess(df):
    """Clean and smooth the spectrum."""
    wl = df["wavelength_nm"]. values
    ab = df["absorbance"].values

    # Clip negative values
    ab = np.clip(ab, 0, None)

    # Smooth using Savitsky-Golay filter
    if len(ab) >= 11:
        window = min(11, len(ab) if len(ab) % 2 != 0 else len(ab) -1)
        ab = savgol_filter(ab, window_length=window, polyorder=3)
        ab = np.clip(ab, 0, None)

    return wl, ab

def detect_peaks(wl, ab, min_prominence=0.05):
    """ Detect peaks and return their positions and absorbances."""
    max_abs = ab.max()
    min_height = max_abs * 0.1

    peak_indices, properties = find_peaks(
        ab,
        prominence=min_prominence,
        height=min_height
    )

    peaks = []
    for i, idx in enumerate(peak_indices):
        prominence = properties["prominences"][i]
        peak_type = "major" if prominence >= 0.1 else "minor"
        peaks.append({
            "wavelength_nm": round(float(wl[idx]), 1),
            "absorbance": round(float(ab[idx]), 4),
            "type": peak_type
        })

    # Sort by absorbance descending
    peaks.sort(key=lambda x: x["absorbance"], reverse=True)
    return peaks

def load_multi_spectra(df):
    """
    Handle a multi-column spectrum CSV.
    First column = wavelength, remaining columns = individual spectra.
    Returns a dict of {column_name: (wavelengths, absorbances)}
    """
    spectra = {}
    wavelengths = df.iloc[:, 0].values

    for col in df.columns[1:]:
        ab = df[col].values.astype(float)
        ab = np.clip(ab, 0, None)
        if len(ab) >=11:
            window = min(11, len (ab) if len(ab) % 2 != 0 else len(ab) - 1)
            ab = savgol_filter(ab, window_length=window, polyorder=3)
            ab = np.clip(ab, 0, None)
        spectra[col] = (wavelengths, ab)

    return spectra