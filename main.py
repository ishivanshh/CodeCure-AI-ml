import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

target_cols = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

# EXACT training schema:
# 5 descriptors + accidental extra 0..4 + fingerprint 0..1023
feature_cols = (
    ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA"] +
    [str(i) for i in range(5)] +
    [str(i) for i in range(1024)]
)

def build_input(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 5 descriptors
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
    ]

    # accidental extra columns 0..4 -> keep zero
    accidental = [0, 0, 0, 0, 0]

    # Morgan fingerprint bits 0..1023
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_bits = list(map(int, list(fp)))

    # exact order as training
    values = descriptors + accidental + fp_bits

    X_input = pd.DataFrame([values], columns=feature_cols)
    return X_input

def predict_toxicity(smiles: str):
    X_input = build_input(smiles)
    if X_input is None:
        return None

    # scale with same scaler used in training
    X_scaled = scaler.transform(X_input)

    # probability-based prediction
    proba = model.predict_proba(X_scaled)

    prediction = []
    for i in range(len(proba)):
        p = proba[i][0][1]  # toxic class probability
        prediction.append(1 if p > 0.25 else 0)

    return {
        target_cols[i]: prediction[i]
        for i in range(len(target_cols))
    }
