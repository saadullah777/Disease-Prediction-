import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Disease Prediction from Medical Data", 
    page_icon="🩺", 
    layout="centered"
)

# Load the trained model
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    with MODEL_PATH.open('rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("Create the model by running the training script from the project folder:")
    st.code(f'python "{MODEL_PATH.parent / "train_model.py"}"', language="powershell")
    st.stop()

# App Header
st.title("🩺 Disease Prediction from Medical Data")
st.write("Predict the possibility of diseases based on patient structured medical parameters[cite: 1].")

# Sidebar inputs for patient data
st.sidebar.header("Patient Input Parameters")

def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose Level', 0, 200, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin Level (mu U/ml)', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.37, 0.01)
    age = st.sidebar.slider('Age (Years)', 21, 81, 29)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Panel Display
st.subheader("Patient Parameter Summary")
st.dataframe(input_df)

# Prediction Button
if st.button("Predict Disease Risk", type="primary"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.error(f"⚠️ **Result:** Positive - High risk of disease detected. Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success(f"✅ **Result:** Negative - Patient is normal / low risk. Confidence: {prediction_proba[0][0]*100:.2f}%")