"""
UV-Vis Training Dataset Builder
Defines the data format and tools for assembling fine-tuning datasets.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────
# Schema: A single training example
# ─────────────────────────────────────────────

@dataclass
class UVVisTrainingExample:
    """
    One training example for the UV-Vis interpretation model.

    - spectrum_text: the preprocessed, structured text description of the spectrum
    - compound_name: known compound name (ground truth)
    - smiles: SMILES string if available
    - molecular_formula: e.g. "C6H6"
    - interpretation: the target output — expert-level interpretation
    - source: where this data came from (SDBS, NIST, literature, etc.)
    - confidence: "high" / "medium" / "low" (quality flag)
    """
    spectrum_text: str
    compound_name: str
    interpretation: str
    smiles: Optional[str] = None
    molecular_formula: Optional[str] = None
    solvent: Optional[str] = None
    source: Optional[str] = None
    confidence: str = "high"


# ─────────────────────────────────────────────
# Interpretation output schema
# ─────────────────────────────────────────────

INTERPRETATION_TEMPLATE = """\
Compound: {compound_name}
Molecular formula: {molecular_formula}

λmax assignments:
{lambda_assignments}

Chromophore analysis:
{chromophore_analysis}

Compound class inference:
{compound_class}

Additional notes:
{additional_notes}
"""

# Example of a filled interpretation:
EXAMPLE_INTERPRETATION = INTERPRETATION_TEMPLATE.format(
    compound_name="Acetophenone",
    molecular_formula="C8H8O",
    lambda_assignments=(
        "- 243 nm: π→π* transition of the conjugated phenyl-carbonyl system\n"
        "- 320 nm (weak): n→π* transition of the carbonyl group"
    ),
    chromophore_analysis=(
        "Contains an aryl ketone chromophore (phenyl ring conjugated with C=O). "
        "The strong π→π* band at 243 nm is characteristic of this system. "
        "The weak n→π* band near 320 nm is typical of carbonyl compounds and "
        "is symmetry-forbidden, hence the low absorbance."
    ),
    compound_class="Aryl ketone (phenone). Consistent with acetophenone or similar aryl methyl ketone.",
    additional_notes=(
        "Spectrum recorded in ethanol. Solvent effects: polar solvents cause a "
        "hypsochromic shift of the n→π* band and a bathochromic shift of the π→π* band."
    )
)


# ─────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert analytical chemist specializing in UV-Vis spectroscopy. \
Your role is to interpret UV-Vis absorption spectra and provide detailed, \
accurate chemical analysis.

When given a spectrum summary, you will:
1. Assign each absorption band to its electronic transition type (π→π*, n→π*, charge transfer, d-d, etc.)
2. Identify the chromophore(s) responsible for each band
3. Infer the likely compound class or functional groups present
4. Note any solvent effects, fine structure, or unusual features
5. Apply Woodward-Fieser rules where applicable

Be precise with wavelengths, use correct photochemical terminology, \
and clearly distinguish between confident assignments and tentative ones.
"""

INTERPRETATION_PROMPT_TEMPLATE = """\
Please interpret the following UV-Vis spectrum:

{spectrum_text}

{optional_context}

Provide:
1. Assignment of each absorption band (transition type and responsible chromophore)
2. Chromophore analysis
3. Compound class inference (if possible)
4. Any notable features or caveats
"""

IDENTIFICATION_PROMPT_TEMPLATE = """\
A UV-Vis spectrum has been recorded for an unknown compound with the following characteristics:

{spectrum_text}

Additional information:
- Molecular formula: {molecular_formula}
- Solvent: {solvent}

Based on this UV-Vis data:
1. What functional groups or chromophores are likely present?
2. What compound classes are consistent with this spectrum?
3. Are there any features that help narrow down or rule out specific structures?
4. What additional spectroscopic data (NMR, IR, MS) would best complement this UV-Vis spectrum?
"""

WOODWARD_FIESER_PROMPT = """\
Using Woodward-Fieser rules, predict or verify the λmax for the following compound:

Compound: {compound_name}
SMILES: {smiles}
Observed λmax: {observed_lambda_max} nm

Observed spectrum:
{spectrum_text}

Please:
1. Apply the appropriate Woodward-Fieser rules (diene or enone system)
2. Calculate the predicted λmax step by step
3. Compare the predicted value to the observed λmax
4. Comment on any discrepancy
"""


# ─────────────────────────────────────────────
# Build a fine-tuning record (OpenAI / HuggingFace chat format)
# ─────────────────────────────────────────────

def build_chat_record(example: UVVisTrainingExample, task: str = "interpret") -> dict:
    """
    Convert a UVVisTrainingExample into a chat-format training record.

    task options:
        "interpret"   — general interpretation
        "identify"    — unknown compound identification
        "woodward"    — Woodward-Fieser rule application
    """
    optional_context = ""
    if example.solvent:
        optional_context += f"Solvent: {example.solvent}\n"
    if example.smiles:
        optional_context += f"SMILES: {example.smiles}\n"
    if example.molecular_formula:
        optional_context += f"Molecular formula: {example.molecular_formula}\n"

    if task == "interpret":
        user_content = INTERPRETATION_PROMPT_TEMPLATE.format(
            spectrum_text=example.spectrum_text,
            optional_context=optional_context.strip()
        )
    elif task == "identify":
        user_content = IDENTIFICATION_PROMPT_TEMPLATE.format(
            spectrum_text=example.spectrum_text,
            molecular_formula=example.molecular_formula or "Unknown",
            solvent=example.solvent or "Unknown"
        )
    elif task == "woodward":
        user_content = WOODWARD_FIESER_PROMPT.format(
            compound_name=example.compound_name,
            smiles=example.smiles or "Not provided",
            observed_lambda_max="See spectrum",
            spectrum_text=example.spectrum_text
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": example.interpretation}
        ],
        "metadata": {
            "compound": example.compound_name,
            "source": example.source,
            "confidence": example.confidence,
            "task": task
        }
    }


def save_dataset(examples: list[UVVisTrainingExample], output_path: str, task: str = "interpret"):
    """
    Save a list of training examples as a JSONL file (standard fine-tuning format).
    Each line is one training record.
    """
    with open(output_path, "w") as f:
        for example in examples:
            record = build_chat_record(example, task=task)
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(examples)} examples to {output_path}")


# ─────────────────────────────────────────────
# Demo: Build a minimal example dataset
# ─────────────────────────────────────────────

DEMO_EXAMPLES = [
    UVVisTrainingExample(
        spectrum_text="""\
Compound ID: acetophenone_001
Spectral region: UV-Vis (range: 200–400 nm)
Primary absorption maximum (λmax): 243.0 nm (Absorbance = 0.921)
Baseline absorbance: 0.003

Detected peaks:
  1. λ = 243.0 nm, A = 0.921 [major, FWHM ≈ 28.0 nm]
  2. λ = 319.0 nm, A = 0.084 [minor, FWHM ≈ 35.0 nm]

Observations:
  - Shoulder(s) detected: may indicate vibronic fine structure or overlapping transitions.""",
        compound_name="Acetophenone",
        smiles="CC(=O)c1ccccc1",
        molecular_formula="C8H8O",
        solvent="ethanol",
        interpretation="""\
Compound: Acetophenone
Molecular formula: C8H8O

λmax assignments:
- 243 nm (A = 0.921, strong): π→π* transition of the conjugated phenyl-carbonyl (aryl ketone) chromophore
- 319 nm (A = 0.084, weak): n→π* transition of the carbonyl lone pair electrons

Chromophore analysis:
Acetophenone contains an aryl ketone chromophore in which the phenyl ring is conjugated with the carbonyl group. This extended conjugation red-shifts the aromatic π→π* transition from ~254 nm (benzene) to ~243 nm. The weak n→π* band at 319 nm is symmetry-forbidden (low ε ~50 L·mol⁻¹·cm⁻¹), which is consistent with the low absorbance observed.

Compound class inference:
Aryl ketone (phenone). The dual-band pattern — strong UV band near 240–250 nm and weak band near 310–330 nm — is highly characteristic of acetophenone and structurally similar aryl methyl ketones.

Additional notes:
Spectrum recorded in ethanol. In polar solvents, the n→π* band undergoes a hypsochromic (blue) shift relative to non-polar solvents due to hydrogen bonding stabilizing the ground state. The π→π* band conversely undergoes a slight bathochromic (red) shift in polar solvents.""",
        source="SDBS",
        confidence="high"
    ),

    UVVisTrainingExample(
        spectrum_text="""\
Compound ID: benzene_001
Spectral region: UV (range: 200–350 nm)
Primary absorption maximum (λmax): 254.0 nm (Absorbance = 0.043)
Baseline absorbance: 0.001

Detected peaks:
  1. λ = 204.0 nm, A = 0.890 [major, FWHM ≈ 12.0 nm]
  2. λ = 254.0 nm, A = 0.043 [minor, FWHM ≈ 8.0 nm]

Observations:
  - Shoulder(s) detected: may indicate vibronic fine structure.""",
        compound_name="Benzene",
        smiles="c1ccccc1",
        molecular_formula="C6H6",
        solvent="hexane",
        interpretation="""\
Compound: Benzene
Molecular formula: C6H6

λmax assignments:
- 204 nm (A = 0.890, strong): E₁ band (π→π* allowed transition, ε ~7,400 L·mol⁻¹·cm⁻¹)
- 254 nm (A = 0.043, weak): B band (π→π* symmetry-forbidden transition, ε ~250 L·mol⁻¹·cm⁻¹); vibronic fine structure visible as shoulders

Chromophore analysis:
Benzene's UV spectrum is governed entirely by its aromatic π system. The highly symmetry-forbidden B band at 254 nm exhibits characteristic vibronic fine structure (multiple sharp sub-bands), which is a hallmark of benzene and simple benzene derivatives. The stronger E₁ band at 204 nm corresponds to a fully allowed transition.

Compound class inference:
Simple aromatic hydrocarbon (benzene or very lightly substituted benzene). The fine structure at 254 nm and absence of any visible-range absorption is consistent with an unsubstituted or weakly substituted benzene ring with no additional chromophores.

Additional notes:
The vibronic fine structure at 254 nm is diagnostic for benzene in non-polar solvents like hexane. In polar solvents, this fine structure is partially washed out. Addition of electron-donating or withdrawing substituents would cause bathochromic shifts and intensity changes in both bands.""",
        source="NIST",
        confidence="high"
    )
]


if __name__ == "__main__":
    # Show example training record
    record = build_chat_record(DEMO_EXAMPLES[0], task="interpret")
    print("=== EXAMPLE TRAINING RECORD ===\n")
    print(json.dumps(record, indent=2))

    # Save demo dataset
    save_dataset(DEMO_EXAMPLES, "/tmp/uvvis_demo_dataset.jsonl", task="interpret")
