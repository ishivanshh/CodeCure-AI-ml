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

# Exact feature columns used in training
feature_cols = (
    ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA"] +
    ["0", "1", "2", "3", "4"] +
    [f"{i}.1" for i in range(1024)]
)

def build_input(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    row = {col: 0 for col in feature_cols}

    # descriptors
    row["MolWt"] = Descriptors.MolWt(mol)
    row["LogP"] = Descriptors.MolLogP(mol)
    row["NumHDonors"] = Descriptors.NumHDonors(mol)
    row["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
    row["TPSA"] = Descriptors.TPSA(mol)

    # keep accidental extra columns as 0
    for extra_col in ["0", "1", "2", "3", "4"]:
        row[extra_col] = 0

    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_bits = list(fp)

    for i, bit in enumerate(fp_bits):
        row[f"{i}.1"] = int(bit)

    X_input = pd.DataFrame([[row[col] for col in feature_cols]], columns=feature_cols)
    return X_input

def predict_toxicity(smiles: str):
    X_input = build_input(smiles)
    if X_input is None:
        return None

    X_input = scaler.transform(X_input)
    proba = model.predict_proba(X_input)

    prediction = []
    for i in range(len(proba)):
        p = proba[i][0][1]
        prediction.append(1 if p > 0.25 else 0)

    return {
        target_cols[i]: prediction[i]
        for i in range(len(target_cols))
    }
