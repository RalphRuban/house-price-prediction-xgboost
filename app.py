import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

# =========================
# Load and preprocess data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Bangalore.csv")
    
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace(".", "", regex=False)
    
    df = df.dropna()
    
    df["Price_per_sqft"] = df["Price"] / df["Area"]
    
    return df

df = load_data()

# =========================
# Prepare features
# =========================
y = np.log(df["Price"])
X = df.select_dtypes(include=[np.number]).drop(["Price", "Price_per_sqft"], axis=1)

# =========================
# Train model
# =========================
@st.cache_resource
def train_model(X, y):
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
    model.fit(X, y)
    return model

import joblib

model = joblib.load("model.pkl")
feature_columns = joblib.load("features.pkl")

# =========================
# SHAP Explainer
# =========================
explainer = shap.Explainer(model)

# =========================
# UI
# =========================
st.title("🏠 House Price Prediction + Explainability")

st.write("Enter house details:")

area = st.number_input("Area (sqft)", value=1000)
bedrooms = st.number_input("No of Bedrooms", value=2)

# =========================
# Prepare input
# =========================
input_data = pd.DataFrame({
    "Area": [area],
    "No_of_Bedrooms": [bedrooms]
})

# Fill missing columns
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X.columns]

# =========================
# Prediction
# =========================
if st.button("Predict Price"):

    pred_log = model.predict(input_data)
    prediction = np.exp(pred_log)

    st.success(f"💰 Estimated Price: ₹ {int(prediction[0]):,}")

    # =========================
    # SHAP Explanation
    # =========================
    st.subheader("🔍 Why this prediction?")

    shap_values = explainer(input_data)

    # Waterfall plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)