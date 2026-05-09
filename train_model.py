import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor

# Load data
df = pd.read_csv("data/Bangalore.csv")

# Clean columns
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace(".", "", regex=False)

df = df.dropna()

# Feature engineering
df["Price_per_sqft"] = df["Price"] / df["Area"]

# Target
y = np.log(df["Price"])

# Features
X = df.select_dtypes(include=[np.number]).drop(["Price", "Price_per_sqft"], axis=1)

# Train model
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

# Save feature columns (IMPORTANT)
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ Model and features saved!")