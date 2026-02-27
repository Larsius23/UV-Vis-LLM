"""
Built-in UV-Vis compound database.
Contains known λmax values, transition types, and chromophore info
for common organic compounds.
"""

COMPOUND_DATABASE = {
    "benzene": {
        "name": "Benzene",
        "formula": "C6H6",
        "smiles": "c1ccccc1",
        "peaks": [
            {"wavelength_nm": 204, "transition": "π→π* (E1 band)", "epsilon": 7400, "intensity": "strong"},
            {"wavelength_nm": 254, "transition": "π→π* (B band, forbidden)", "epsilon": 250, "intensity": "weak"},
        ],
        "compound_class": "Aromatic hydrocarbon",
        "chromophore": "Benzene ring (aromatic π system)",
        "notes": "fine vibronic structure visible at 254 nm in non-polar solvents."
    },
    "acetophenone": {
        "name": "Acetophenone",
        "formula": "C8H8O",
        "smiles": "CC(=O)c1ccccc1",
        "peaks": [
            {"wavelength_nm": 243, "transition": "π→π* (aryl ketone)", "epsilon": 13000, "intensity": "strong"},
            {"wavelength_nm": 319, "transition": "π→π* (carbonyl)", "epsilon": 50, "intensity": "weak"},
        ],
        "compound_class": "Aryl ketone",
        "chromophore": "Phenyl-carbonyl conjugated system",
        "notes": "dual band pattern is diagnostic for aryl ketones."
    },
    "naphthalene": {
        "name": "Naphthalene",
        "formula": "C10H8",
        "smiles": "c1ccc2ccccc2c1",
        "peaks": [
            {"wavelength_nm": 220, "transition": "π→π* (allowed)", "epsilon": 112000, "intensity": "strong"},
            {"wavelength_nm": 275, "transition": "π→π* (B band)", "epsilon": 5600, "intensity": "medium"},
            {"wavelength_nm": 311, "transition": "π→π* (forbidden)", "epsilon": 250, "intensity": "weak"},
        ],
        "compound_class": "Polycyclic aromatic hydrocarbon",
        "chromophore": "Fused bicyclic aromatic π system",
        "notes": "Extended conjugation red-shifts all bands compared to benzene."
    },
    "anthracene": {
        "name": "Anthracene",
        "formula": "C14H10",
        "smiles": "c1ccc2cc3ccccc3cc2c1",
        "peaks": [
            {"wavelength_nm": 253, "transition": "π→π* (allowed)", "epsilon": 180000, "intensity": "strong"},
            {"wavelength_nm": 375, "transition": "π→π* (B band)", "epsilon": 7900, "intensity": "medium"},
        ],
        "compound_class": "Polycyclic aromatic hydrocarbon",
        "chromophore": "Fused tricyclic aromatic π system",
        "notes": "Strong visible absorption makes anthracene appear pale yellow."
    },
    "caffeine": {
        "name": "Caffeine",
        "formula": "C8H10N4O2",
        "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "peaks": [
            {"wavelength_nm": 273, "transition": "π→π* (purine ring)", "epsilon": 9800, "intensity": "strong"},
        ],
        "compound_class": "Xanthine alkaloid",
        "chromophore": "Purine ring system",
        "notes": "Single strong band at 273 nm is highly characteristic of caffeine."
    },
    "acetone": {
        "name": "Acetone",
        "formula": "C3H6O",
        "smiles": "CC(C)=O",
        "peaks": [
            {"wavelength_nm": 166, "transition": "π→π* (carbonyl)", "epsilon": 16000, "intensity": "strong"},
            {"wavelength_nm": 279, "transition": "n→π* (carbonyl)", "epsilon": 15, "intensity": "very weak"},
        ],
        "compound_class": "Ketone",
        "chromophore": "Carbonyl group (C=O)",
        "notes": "The n→π* band at 279 nm is very weak. Primary π→π* is in the far UV."
    },
    "aniline": {
        "name": "Aniline",
        "formula": "C6H7N",
        "smiles": "Nc1ccccc1",
        "peaks": [
            {"wavelength_nm": 230, "transition": "π→π* (E2 band)", "epsilon": 8600, "intensity": "strong"},
            {"wavelength_nm": 280, "transition": "π→π* (B band)", "epsilon": 1430, "intensity": "medium"},
        ],
        "compound_class": "Aromatic amine",
        "chromophore": "Aniline chromophore (amino-substituted benzene)",
        "notes": "NH2 group donates electrons into ring, causing bathochromic shift vs benzene."
    },
    "phenol": {
        "name": "Phenol",
        "formula": "C6H6O",
        "smiles": "Oc1ccccc1",
        "peaks": [
            {"wavelength_nm": 210, "transition": "π→π* (E1 band)", "epsilon": 6200, "intensity": "strong"},
            {"wavelength_nm": 270, "transition": "π→π* (B band)", "epsilon": 1450, "intensity": "medium"},
        ],
        "compound_class": "Phenol",
        "chromophore": "Hydroxyl-substituted benzene ring",
        "notes": "OH group causes bathochromic shift compared to benzene. Shifts further in alkaline solution."
    },
    "toluene": {
        "name": "Toluene",
        "formula": "C7H8",
        "smiles": "Cc1ccccc1",
        "peaks": [
            {"wavelength_nm": 206, "transition": "π→π* (E1 band)", "epsilon": 7000, "intensity": "strong"},
            {"wavelength_nm": 262, "transition": "π→π* (B band)", "epsilon": 300, "intensity": "weak"},
        ],
        "compound_class": "Alkyl-substituted aromatic",
        "chromophore": "Methyl-substituted benzene ring",
        "notes": "Small bathochromic shift vs benzene due to methyl hyperconjugation."
    },
    "nitrobenzene": {
        "name": "Nitrobenzene",
        "formula": "C6H5NO2",
        "smiles": "O=[N+]([O-])c1ccccc1",
        "peaks": [
            {"wavelength_nm": 252, "transition": "π→π*", "epsilon": 10000, "intensity": "strong"},
            {"wavelength_nm": 330, "transition": "n→π* (nitro group)", "epsilon": 125, "intensity": "weak"},
        ],
        "compound_class": "Nitroaromatic compound",
        "chromophore": "Nitro-substituted benzene ring",
        "notes": "Strong electron-withdrawing nitro group causes significant bathochromic shift."
    }
}


def search_compound(query: str) -> list:
    """
    Search the database for compounds matching the query.
    Returns a list of matching compound dicts.
    """
    query = query.lower().strip()
    results = []

    for key, compound in COMPOUND_DATABASE.items():
        if (query in key or
            query in compound["name"].lower() or
            query in compound["formula"].lower() or
            query in compound["compound_class"].lower()):
            results.append(compound)

    return results


def get_all_compounds() -> list:
    """Return all compound names for the dropdown."""
    return [c["name"] for c in COMPOUND_DATABASE.values()]