# UV-Vis Spectroscopy LLM Prompt Library
# ==========================================
# A curated set of prompts for different UV-Vis interpretation tasks.
# Use these for zero-shot / few-shot inference, or as the basis for fine-tuning.

# ─────────────────────────────────────────────
# SYSTEM PROMPT (use for all tasks)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert analytical chemist specializing in UV-Vis spectroscopy with deep knowledge of:
- Electronic transition theory (π→π*, n→π*, charge transfer, d-d transitions)
- Chromophore theory and functional group UV absorption
- Woodward-Fieser rules for dienes, enones, and aromatic systems
- Solvatochromism and solvent effects on absorption bands
- Beer-Lambert law and quantitative UV-Vis analysis

When interpreting UV-Vis spectra, always:
1. Assign each detected band to a specific electronic transition type
2. Identify the chromophore responsible for each transition
3. Cite the approximate molar absorptivity (ε) range expected for that transition type
4. Distinguish clearly between high-confidence assignments and tentative ones
5. Note any features that are unusual, diagnostic, or structurally informative

Respond with structured, precise chemical language. When uncertain, say so explicitly.
""".strip()


# ─────────────────────────────────────────────
# PROMPT 1: General interpretation
# ─────────────────────────────────────────────

GENERAL_INTERPRETATION_PROMPT = """
Interpret the following UV-Vis spectrum:

{spectrum_text}

Provide a complete interpretation including:
1. **Band assignments**: For each detected peak or shoulder, identify the transition type (π→π*, n→π*, charge transfer, etc.) and the responsible chromophore
2. **Chromophore analysis**: Describe the electronic structure giving rise to the observed absorptions
3. **Compound class**: What functional groups or compound classes are consistent with this spectrum?
4. **Diagnostic features**: Are there any features (fine structure, band ratios, unusual positions) that are particularly informative?
5. **Caveats**: Note anything that is ambiguous or would require additional data to resolve
""".strip()


# ─────────────────────────────────────────────
# PROMPT 2: Unknown compound identification
# ─────────────────────────────────────────────

UNKNOWN_IDENTIFICATION_PROMPT = """
An unknown organic compound has been analyzed by UV-Vis spectroscopy. Help identify it.

UV-Vis spectrum:
{spectrum_text}

Known information:
- Molecular formula: {molecular_formula}
- Degree of unsaturation: {degree_of_unsaturation}
- Solvent: {solvent}

Based on this UV-Vis data:
1. What chromophores are present? Justify each assignment with the observed λmax and approximate ε
2. What compound classes are consistent with the full spectral profile?
3. Which compound classes can be ruled out and why?
4. Rank the 2–3 most likely compound classes by probability
5. What complementary spectroscopic technique (IR, NMR, MS) would most efficiently disambiguate between your top candidates?
""".strip()


# ─────────────────────────────────────────────
# PROMPT 3: Woodward-Fieser rules
# ─────────────────────────────────────────────

WOODWARD_FIESER_DIENE_PROMPT = """
Apply Woodward-Fieser rules to predict and verify the λmax for the following diene system.

Compound: {compound_name}
Structure description: {structure_description}
SMILES: {smiles}

Observed UV-Vis spectrum:
{spectrum_text}

Steps required:
1. Identify the diene system type (s-cis or s-trans, homoannular or heteroannular)
2. Start with the base value (253 nm for homoannular; 217 nm for heteroannular)
3. Add increments for: each extending double bond (+30 nm), each alkyl substituent or ring residue (+5 nm), each exocyclic double bond (+5 nm), each polar group (+5 to +17 nm depending on type)
4. State the predicted λmax
5. Compare to observed λmax and comment on agreement or discrepancy
""".strip()


WOODWARD_FIESER_ENONE_PROMPT = """
Apply Woodward-Fieser rules to predict and verify the λmax for the following α,β-unsaturated carbonyl system.

Compound: {compound_name}
Structure description: {structure_description}
SMILES: {smiles}

Observed UV-Vis spectrum:
{spectrum_text}

Steps required:
1. Identify the carbonyl type (aldehyde, ketone, acid, ester) and base value
   - α,β-unsaturated aldehyde: base 208 nm
   - α,β-unsaturated ketone (acyclic or 6-membered ring): base 215 nm
   - α,β-unsaturated ketone (5-membered ring): base 202 nm
   - α,β-unsaturated acid/ester: base 193 nm
2. Add increments for: α-substituents (+10 nm each), β-substituents (+12 nm each), double bond extending conjugation (+30 nm), exocyclic double bond (+5 nm), homoannular diene (+39 nm)
3. State the predicted λmax
4. Compare to observed λmax and comment on agreement or discrepancy
""".strip()


# ─────────────────────────────────────────────
# PROMPT 4: Concentration / Beer-Lambert
# ─────────────────────────────────────────────

BEER_LAMBERT_PROMPT = """
Help analyze this UV-Vis measurement using Beer-Lambert law.

Compound: {compound_name}
Molar absorptivity (ε) at λmax: {epsilon} L·mol⁻¹·cm⁻¹
Path length: {path_length} cm
Observed spectrum:
{spectrum_text}

Please:
1. Calculate the concentration of the sample using A = εlc
2. Comment on whether the absorbance value is in the reliable linear range (ideally 0.1 < A < 1.0)
3. If the absorbance is outside this range, suggest how to adjust (dilution factor or path length change)
4. Note any deviations from Beer-Lambert behavior (non-linearity, stray light, solute-solute interactions)
""".strip()


# ─────────────────────────────────────────────
# PROMPT 5: Solvent effect analysis
# ─────────────────────────────────────────────

SOLVENT_EFFECT_PROMPT = """
Analyze the solvent effect on the following UV-Vis spectrum.

Compound: {compound_name}
Solvent used: {solvent}
Spectrum:
{spectrum_text}

Reference data (if available):
{reference_data}

Please:
1. Identify which bands are most sensitive to solvent polarity
2. Predict whether the observed λmax positions would shift bathochromically (red shift) or hypsochromically (blue shift) in a more polar / less polar solvent — and why
3. For n→π* transitions specifically: explain the expected solvent effect direction and mechanism
4. For π→π* transitions: explain the expected solvent effect direction and mechanism
5. Is the solvent choice appropriate for this type of compound?
""".strip()


# ─────────────────────────────────────────────
# PROMPT 6: Mixture / purity analysis
# ─────────────────────────────────────────────

MIXTURE_ANALYSIS_PROMPT = """
A UV-Vis spectrum has been recorded for what may be a mixture or an impure sample.

Spectrum:
{spectrum_text}

Expected major component: {expected_compound}
Expected λmax of major component: {expected_lambda_max} nm

Please:
1. Identify features that are consistent with the expected compound
2. Identify any peaks, shoulders, or baseline features that are inconsistent with the expected compound and may indicate an impurity or second component
3. If an impurity is suspected, suggest what class of compound it might belong to based on the unexpected absorption
4. Estimate whether the sample appears pure or contaminated, and to what degree
5. Recommend the best approach to confirm purity (e.g., baseline subtraction, chromatographic separation + re-measurement)
""".strip()


# ─────────────────────────────────────────────
# FEW-SHOT EXAMPLES (attach to any prompt above)
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
---
EXAMPLE 1:

Input spectrum:
Compound ID: example_001
Spectral region: UV-Vis (range: 200–500 nm)
Primary absorption maximum (λmax): 243.0 nm (Absorbance = 0.921)
Detected peaks:
  1. λ = 243.0 nm, A = 0.921 [major, FWHM ≈ 28 nm]
  2. λ = 319.0 nm, A = 0.084 [minor, FWHM ≈ 35 nm]

Expert interpretation:
- 243 nm: π→π* transition of the aryl ketone chromophore (phenyl ring conjugated with C=O). Strong band, ε ~13,000 L·mol⁻¹·cm⁻¹.
- 319 nm: n→π* transition of the carbonyl group. Weak and broad (symmetry-forbidden), ε ~50 L·mol⁻¹·cm⁻¹.
- Compound class: Aryl ketone. Pattern is diagnostic for acetophenone-type structures.

---
EXAMPLE 2:

Input spectrum:
Compound ID: example_002
Spectral region: UV (range: 200–350 nm)
Primary absorption maximum (λmax): 204.0 nm (Absorbance = 0.890)
Detected peaks:
  1. λ = 204.0 nm, A = 0.890 [major, FWHM ≈ 12 nm]
  2. λ = 254.0 nm, A = 0.043 [minor, FWHM ≈ 8 nm]
Observations: Vibronic fine structure visible near 254 nm.

Expert interpretation:
- 204 nm: E₁ band, allowed π→π* transition of the benzene ring. ε ~7,400 L·mol⁻¹·cm⁻¹.
- 254 nm: B band, symmetry-forbidden π→π* transition. ε ~250 L·mol⁻¹·cm⁻¹. Vibronic fine structure is diagnostic for unsubstituted benzene.
- Compound class: Simple aromatic hydrocarbon (benzene or very lightly substituted derivative).
---
""".strip()


# ─────────────────────────────────────────────
# Helper: build a complete prompt with few-shot
# ─────────────────────────────────────────────

def build_prompt(
    spectrum_text: str,
    task: str = "general",
    include_few_shot: bool = True,
    **kwargs
) -> dict:
    """
    Build a complete prompt dict ready for API submission.

    task options: "general", "identify", "woodward_diene", "woodward_enone",
                  "beer_lambert", "solvent", "mixture"
    """
    PROMPT_MAP = {
        "general":         GENERAL_INTERPRETATION_PROMPT,
        "identify":        UNKNOWN_IDENTIFICATION_PROMPT,
        "woodward_diene":  WOODWARD_FIESER_DIENE_PROMPT,
        "woodward_enone":  WOODWARD_FIESER_ENONE_PROMPT,
        "beer_lambert":    BEER_LAMBERT_PROMPT,
        "solvent":         SOLVENT_EFFECT_PROMPT,
        "mixture":         MIXTURE_ANALYSIS_PROMPT,
    }

    template = PROMPT_MAP.get(task, GENERAL_INTERPRETATION_PROMPT)
    user_content = template.format(spectrum_text=spectrum_text, **kwargs)

    if include_few_shot:
        user_content = FEW_SHOT_EXAMPLES + "\n\nNow interpret the following:\n\n" + user_content

    return {
        "system": SYSTEM_PROMPT,
        "user": user_content
    }


if __name__ == "__main__":
    # Quick demo
    sample_spectrum = """
Compound ID: unknown_sample
Spectral region: UV-Vis (range: 200–600 nm)
Primary absorption maximum (λmax): 372.0 nm (Absorbance = 1.340)
Baseline absorbance: 0.008

Detected peaks:
  1. λ = 372.0 nm, A = 1.340 [major, FWHM ≈ 45 nm]
  2. λ = 252.0 nm, A = 0.780 [major, FWHM ≈ 22 nm]
  3. λ = 430.0 nm, A = 0.210 [minor, FWHM ≈ 38 nm]

Observations:
  - Multiple major peaks detected: may indicate a mixture or complex chromophore system.
""".strip()

    prompt = build_prompt(sample_spectrum, task="general", include_few_shot=True)
    print("=== SYSTEM ===")
    print(prompt["system"])
    print("\n=== USER ===")
    print(prompt["user"])
