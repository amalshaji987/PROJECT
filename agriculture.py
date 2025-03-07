import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("agriculture_data.csv")
    return df

df = load_data()

# Train Model
def train_model():
    X = df[['Temperature', 'Rainfall', 'Soil_Quality', 'Fertilizer_Used', 'Pesticide_Used']]
    y = df['Yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "crop_yield_model.pkl")
    return model

# Load or Train Model
try:
    model = joblib.load("crop_yield_model.pkl")
except:
    model = train_model()

# Streamlit UI
st.title("AI-Powered Crop Yield Prediction")
st.write("Enter the environmental factors below to predict crop yield:")

temp = st.slider("Temperature (°C)", 15.0, 35.0, 25.0)
rain = st.slider("Rainfall (mm)", 200.0, 1200.0, 600.0)
soil = st.slider("Soil Quality (1-10)", 1.0, 10.0, 5.0)
fertilizer = st.slider("Fertilizer Used (kg/hectare)", 50.0, 300.0, 150.0)
pesticide = st.slider("Pesticide Used (kg/hectare)", 1.0, 10.0, 5.0)

if st.button("Predict Crop Yield"):
    input_data = np.array([[temp, rain, soil, fertilizer, pesticide]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Crop Yield: {prediction[0]:.2f} tons/hectare")

st.write("### Dataset Sample")
st.dataframe(df.head())