import streamlit as st
import joblib
import pandas as pd
import os

# Build absolute path to models folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define feature names (must match training)
FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

# Map class indices to species names
CLASS_NAMES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# Prediction function
def predict_iris(features):
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)[0]
    return CLASS_NAMES[prediction]

# Streamlit UI
st.title("🌸 Iris Flower Classification")
st.write("Enter flower measurements to predict the species:")
st.caption("ℹ️ After typing a value, press **Enter** to confirm it.")   # <- global hint

sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    sample = [sepal_length, sepal_width, petal_length, petal_width]
    species = predict_iris(sample)
    st.success(f"✅ Predicted species: **{species}**")
