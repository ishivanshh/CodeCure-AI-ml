import streamlit as st
from main import predict_toxicity

st.title("🧪 Toxicity Predictor")

# 👇 YEH PART ADD KARO
st.markdown("""
### ℹ️ About This App
- **SMILES** is a way to represent chemical structures using text  
- Enter a valid SMILES string to predict toxicity  
- The model predicts across **12 biological endpoints**  
- Output shows whether a compound is **Toxic or Non-Toxic**
""")

# Optional: example inputs
st.markdown("""
**Example SMILES you can try:**
- CCO (ethanol)  
- c1ccccc1 (benzene)  
- CC(=O)O (acetic acid)
""")

# 👇 existing input
smiles = st.text_input("Enter SMILES:")

if st.button("Predict"):
    result = predict_toxicity(smiles)

    if result is None:
        st.error("Invalid SMILES")
    else:
        st.subheader("Prediction Results")
        for label, value in result.items():
            st.write(f"**{label}**: {'Toxic' if value == 1 else 'Non-Toxic'}")
