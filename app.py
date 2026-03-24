import streamlit as st
from main import predict_toxicity

st.title("🧪 Toxicity Predictor")

smiles = st.text_input("Enter SMILES:")

if st.button("Predict"):
    result = predict_toxicity(smiles)

    if result is None:
        st.error("Invalid SMILES")
    else:
        st.subheader("Prediction Results")
        for label, value in result.items():
            st.write(f"**{label}**: {'Toxic' if value == 1 else 'Non-Toxic'}")