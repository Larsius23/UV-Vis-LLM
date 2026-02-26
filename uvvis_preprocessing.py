"""
UV-Vis Spectrum Preprocessing Pipeline
Converts raw spectral data into structured, LLM-readable text summaries.
"""

import numpy as np
import json
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, asdict
from typing import Optional
import csv
import os


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Peak:
    wavelength_nm: float
    absorbance: float
    type: str  # "major", "minor", "shoulder"
    fwhm_nm: Optional[float] = None  # Full width at half maximum

@dataclass
class SpectrumSummary:
    compound_id: str
    wavelength_range: tuple
    lambda_max: float             # Primary peak wavelength
    absorbance_max: float         # Absorbance at lambda_max
    peaks: list[Peak]
    baseline_absorbance: float
    spectral_region: str          # UV / Vis / UV-Vis
    notes: list[str]              # Any flags or observations


# ─────────────────────────────────────────────
# Step 1: Load spectrum from CSV
# ─────────────────────────────────────────────

def load_spectrum(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a UV-Vis spectrum from a two-column CSV (wavelength, absorbance).
    Returns wavelength and absorbance arrays, sorted by wavelength.
    """
    wavelengths, absorbances = [], []

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0 and not _is_numeric(row[0]):
                continue  # skip header
            wavelengths.append(float(row[0]))
            absorbances.append(float(row[1]))

    wl = np.array(wavelengths)
    ab = np.array(absorbances)

    # Sort by wavelength ascending
    sort_idx = np.argsort(wl)
    return wl[sort_idx], ab[sort_idx]


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ─────────────────────────────────────────────
# Step 2: Clean and normalize
# ─────────────────────────────────────────────

def preprocess_spectrum(
    wavelengths: np.ndarray,
    absorbances: np.ndarray,
    smooth: bool = True,
    clip_negative: bool = True,
    normalize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean the raw spectrum:
    - Clip negative absorbance values (instrument noise artifacts)
    - Smooth using Savitzky-Golay filter
    - Optionally normalize to max absorbance = 1.0
    """
    ab = absorbances.copy()

    if clip_negative:
        ab = np.clip(ab, 0, None)

    if smooth and len(ab) >= 11:
        # window_length must be odd and less than data length
        window = min(11, len(ab) if len(ab) % 2 != 0 else len(ab) - 1)
        ab = savgol_filter(ab, window_length=window, polyorder=3)
        ab = np.clip(ab, 0, None)  # re-clip after smoothing

    if normalize and ab.max() > 0:
        ab = ab / ab.max()

    return wavelengths, ab


# ─────────────────────────────────────────────
# Step 3: Peak detection
# ─────────────────────────────────────────────

def detect_peaks(
    wavelengths: np.ndarray,
    absorbances: np.ndarray,
    min_prominence: float = 0.05,
    min_height_fraction: float = 0.1
) -> list[Peak]:
    """
    Detect peaks and shoulders in the spectrum.
    Returns a list of Peak objects, sorted by absorbance (major peaks first).
    """
    max_abs = absorbances.max()
    min_height = max_abs * min_height_fraction

    # Find standard peaks
    peak_indices, properties = find_peaks(
        absorbances,
        prominence=min_prominence,
        height=min_height
    )

    peaks = []
    for idx in peak_indices:
        wl = round(float(wavelengths[idx]), 1)
        ab = round(float(absorbances[idx]), 4)
        prominence = properties["prominences"][list(peak_indices).index(idx)]

        peak_type = "major" if prominence >= 0.1 else "minor"
        fwhm = _estimate_fwhm(wavelengths, absorbances, idx)

        peaks.append(Peak(
            wavelength_nm=wl,
            absorbance=ab,
            type=peak_type,
            fwhm_nm=fwhm
        ))

    # Detect shoulders using second derivative
    shoulders = _detect_shoulders(wavelengths, absorbances, existing_peaks=peak_indices)
    peaks.extend(shoulders)

    # Sort by absorbance descending
    peaks.sort(key=lambda p: p.absorbance, reverse=True)
    return peaks


def _estimate_fwhm(
    wavelengths: np.ndarray,
    absorbances: np.ndarray,
    peak_idx: int
) -> Optional[float]:
    """Estimate Full Width at Half Maximum for a peak."""
    half_max = absorbances[peak_idx] / 2.0

    # Search left
    left_idx = peak_idx
    while left_idx > 0 and absorbances[left_idx] > half_max:
        left_idx -= 1

    # Search right
    right_idx = peak_idx
    while right_idx < len(absorbances) - 1 and absorbances[right_idx] > half_max:
        right_idx += 1

    if left_idx == 0 or right_idx == len(absorbances) - 1:
        return None  # Can't estimate if peak is at boundary

    fwhm = round(float(wavelengths[right_idx] - wavelengths[left_idx]), 1)
    return fwhm


def _detect_shoulders(
    wavelengths: np.ndarray,
    absorbances: np.ndarray,
    existing_peaks: np.ndarray,
    threshold: float = 0.01
) -> list[Peak]:
    """Detect shoulders using inflection points in the second derivative."""
    shoulders = []

    if len(absorbances) < 5:
        return shoulders

    smoothed = gaussian_filter1d(absorbances, sigma=3)
    second_deriv = np.gradient(np.gradient(smoothed))

    # Find sign changes in second derivative (inflection points)
    sign_changes = np.where(np.diff(np.sign(second_deriv)))[0]

    for idx in sign_changes:
        # Skip if too close to an existing peak
        if any(abs(idx - p) < 5 for p in existing_peaks):
            continue
        # Only report if absorbance is significant
        if absorbances[idx] > absorbances.max() * 0.15:
            shoulders.append(Peak(
                wavelength_nm=round(float(wavelengths[idx]), 1),
                absorbance=round(float(absorbances[idx]), 4),
                type="shoulder"
            ))

    return shoulders


# ─────────────────────────────────────────────
# Step 4: Build structured summary
# ─────────────────────────────────────────────

def build_summary(
    compound_id: str,
    wavelengths: np.ndarray,
    absorbances: np.ndarray,
    peaks: list[Peak]
) -> SpectrumSummary:
    """Assemble a SpectrumSummary from spectrum data and detected peaks."""

    wl_min, wl_max = float(wavelengths.min()), float(wavelengths.max())
    lambda_max_idx = np.argmax(absorbances)
    lambda_max = round(float(wavelengths[lambda_max_idx]), 1)
    abs_max = round(float(absorbances[lambda_max_idx]), 4)
    baseline = round(float(np.percentile(absorbances, 5)), 4)

    # Determine spectral region
    if wl_max < 400:
        region = "UV"
    elif wl_min >= 400:
        region = "Vis"
    else:
        region = "UV-Vis"

    # Auto-generate notes
    notes = []
    if abs_max > 2.0:
        notes.append("High absorbance (>2.0): possible concentration issue or strong chromophore.")
    if baseline > 0.1:
        notes.append("Elevated baseline: possible scattering, turbidity, or reference mismatch.")
    if len([p for p in peaks if p.type == "major"]) > 3:
        notes.append("Multiple major peaks detected: may indicate a mixture or complex chromophore system.")
    if any(p.type == "shoulder" for p in peaks):
        notes.append("Shoulder(s) detected: may indicate vibronic fine structure or overlapping transitions.")

    return SpectrumSummary(
        compound_id=compound_id,
        wavelength_range=(round(wl_min, 1), round(wl_max, 1)),
        lambda_max=lambda_max,
        absorbance_max=abs_max,
        peaks=peaks,
        baseline_absorbance=baseline,
        spectral_region=region,
        notes=notes
    )


# ─────────────────────────────────────────────
# Step 5: Serialize to LLM-readable text
# ─────────────────────────────────────────────

def summary_to_text(summary: SpectrumSummary) -> str:
    """
    Convert a SpectrumSummary into a clean natural language string
    suitable for LLM input.
    """
    lines = [
        f"Compound ID: {summary.compound_id}",
        f"Spectral region: {summary.spectral_region} "
        f"(range: {summary.wavelength_range[0]}–{summary.wavelength_range[1]} nm)",
        f"Primary absorption maximum (λmax): {summary.lambda_max} nm "
        f"(Absorbance = {summary.absorbance_max})",
        f"Baseline absorbance: {summary.baseline_absorbance}",
        "",
        "Detected peaks:"
    ]

    for i, peak in enumerate(summary.peaks, 1):
        fwhm_str = f", FWHM ≈ {peak.fwhm_nm} nm" if peak.fwhm_nm else ""
        lines.append(
            f"  {i}. λ = {peak.wavelength_nm} nm, "
            f"A = {peak.absorbance} [{peak.type}{fwhm_str}]"
        )

    if summary.notes:
        lines.append("")
        lines.append("Observations:")
        for note in summary.notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def summary_to_json(summary: SpectrumSummary) -> str:
    """Serialize summary to JSON (useful for structured fine-tuning datasets)."""
    d = asdict(summary)
    return json.dumps(d, indent=2)


# ─────────────────────────────────────────────
# Full Pipeline Entry Point
# ─────────────────────────────────────────────

def process_spectrum_file(
    filepath: str,
    compound_id: Optional[str] = None,
    output_format: str = "text"  # "text" or "json"
) -> str:
    """
    Full pipeline: load → preprocess → detect peaks → summarize → serialize.

    Args:
        filepath: path to CSV file (wavelength, absorbance)
        compound_id: optional identifier (defaults to filename)
        output_format: "text" for LLM prompt input, "json" for dataset building

    Returns:
        Formatted string summary
    """
    if compound_id is None:
        compound_id = os.path.splitext(os.path.basename(filepath))[0]

    wl, ab = load_spectrum(filepath)
    wl, ab = preprocess_spectrum(wl, ab)
    peaks = detect_peaks(wl, ab)
    summary = build_summary(compound_id, wl, ab, peaks)

    if output_format == "json":
        return summary_to_json(summary)
    return summary_to_text(summary)


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate a spectrum for testing (benzene-like, ~254 nm peak)
    wl = np.linspace(200, 800, 601)
    ab = (
        0.8 * np.exp(-((wl - 254) ** 2) / (2 * 15 ** 2)) +
        0.3 * np.exp(-((wl - 280) ** 2) / (2 * 10 ** 2)) +
        0.05 * np.random.normal(0, 0.01, len(wl))
    )
    ab = np.clip(ab, 0, None)

    # Save to temp CSV
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("wavelength_nm,absorbance\n")
        for w, a in zip(wl, ab):
            f.write(f"{w:.1f},{a:.5f}\n")
        tmp_path = f.name

    result = process_spectrum_file(tmp_path, compound_id="benzene_example", output_format="text")
    print(result)
    print("\n--- JSON FORMAT ---\n")
    result_json = process_spectrum_file(tmp_path, compound_id="benzene_example", output_format="json")
    print(result_json)
