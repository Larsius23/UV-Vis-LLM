def calculate_concentration(absorbance: float, epsilon: float, path_length: float) ->dict:
    """
    Calculate concentration using Beer-Lambert law: A = εlc
    Returns concentration in mol/L and a validity assessment.
    """
    if epsilon <= 0:
        raise ValueError("Molar absorptivity (ε) must be breater than 0.")
    if path_length <= 0:
        raise ValueError("Path length must be greater than 0.")
    if absorbance < 0:
        raise ValueError("Absorbance cannot be negative.")
    
    concentration = absorbance / (epsilon * path_length)

    # Assess reliability
    if absorbance < 0.1:
        reliability = " LOW - absorbance below 0.1, consider increasing concentration or path length."
    elif absorbance > 1.0:
        reliability = "HIGH - absorbance above 1.0, consider diluting the sample."
    else:
        reliability = "GOOD - absorbance is in the ideal range (0.1-1.0)."

    return {
        "concentration": concentration,
        "absorbance": absorbance,
        "epsilon": epsilon,
        "path_length": path_length,
        "reliability": reliability
    }