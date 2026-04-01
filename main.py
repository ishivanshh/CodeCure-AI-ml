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

# Use exact feature names from scaler
feature_cols = list(scaler.feature_names_in_)

def build_input(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Start with all-zero row using EXACT training schema
    row = pd.Series(0.0, index=feature_cols, dtype="float64")

    # Descriptors
    descriptor_map = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
    }

    for col, val in descriptor_map.items():
        if col in row.index:
            row[col] = float(val)

    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_bits = list(map(int, list(fp)))

    for i, bit in enumerate(fp_bits):
        dotted_col = f"{i}.1"
        plain_col = str(i)

        # Prefer exact scaler schema
        if dotted_col in row.index:
            row[dotted_col] = bit
        elif plain_col in row.index:
            row[plain_col] = bit

    X_input = pd.DataFrame([row.values], columns=feature_cols)
    return X_input

def predict_toxicity(smiles: str):
    X_input = build_input(smiles)
    if X_input is None:
        return None

    # Transform using same scaler
    X_scaled = scaler.transform(X_input)

    # Probability-based prediction
    proba = model.predict_proba(X_scaled)

    prediction = []
    for i in range(len(proba)):
        p = proba[i][0][1]  # toxic class probability
        prediction.append(1 if p > 0.25 else 0)

    return {
        target_cols[i]: prediction[i]
        for i in range(len(target_cols))
    }

