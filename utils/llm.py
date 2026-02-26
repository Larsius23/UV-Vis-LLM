import ollama

def interpret_spectrum(peaks: list, wl_range: tuple, lambda_max: float) -> str:
    """
    Send spectrum summary to Llama3 and get a scientific interpretation.
    """

    # Build a readable peak summary
    peak_lines = ""
    for peak in peaks:
        peak_lines += f" - λ {peak['wavelength_nm']} nm, Absorbance = {peak['absorbance']} ({peak['type']} peak)\n"

    prompt = f"""
You are an expert analytical chemist specializing in UV-VIs Spectroscopy.

A UV-Vis spectrum has been recorded with the following characteristics:
- Wavelength range: {wl_range[0]:.0f}-{wl_range[1]:.0f} nm
- Primary λmax: {lambda_max:.1f} nm
- Detected peaks:
{peak_lines}

Please provide:
1. The likely electronic transition type for each peak (π→π*, n→π*, charge transfer, etc.)
2. The chromophore(s) responsible for each absorption band
3. The most likely compound class or functional groups present
4. Any additional observations about the spectrum

Be precise and use correct photochemical terminology.
""".strip()
    
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": "You are an expert analytical chemist specializing in UV-Vis spectroscopy. Give precise, scientific interpretations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]