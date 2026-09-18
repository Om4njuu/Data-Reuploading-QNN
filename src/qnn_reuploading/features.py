#RDKit feature extraction for the QNN input contract

from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

FEATURE_NAMES = (
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "ring_count",
    "heavy_atom_count",
    "fraction_csp3",
    "molar_refractivity",
    "aromatic_ring_count",
)

#return 11 deterministic RDKit descriptors for one SMILES string
def featurize_smiles(smiles: str) -> list[float]:
    molecule = getattr(Chem, "MolFromSmiles")(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return [
        float(Descriptors.MolWt(molecule)),
        float(Crippen.MolLogP(molecule)),
        float(rdMolDescriptors.CalcTPSA(molecule)),
        float(Lipinski.NumHDonors(molecule)),
        float(Lipinski.NumHAcceptors(molecule)),
        float(Lipinski.NumRotatableBonds(molecule)),
        float(rdMolDescriptors.CalcNumRings(molecule)),
        float(molecule.GetNumHeavyAtoms()),
        float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        float(Crippen.MolMR(molecule)),
        float(rdMolDescriptors.CalcNumAromaticRings(molecule)),
    ]

#column-standardize descriptors without adding a preprocessing dependency
def standardize(rows: list[list[float]]) -> list[list[float]]:
    if not rows or any(len(row) != len(FEATURE_NAMES) for row in rows):
        raise ValueError("rows must contain 11-feature vectors")
    columns = list(zip(*rows))
    means = [sum(column) / len(column) for column in columns]
    scales = []
    for column, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        scales.append(variance**0.5 or 1.0)
    return [[(value - mean) / scale for value, mean, scale in zip(row, means, scales)] for row in rows]