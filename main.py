import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("data/Bangalore.csv")

# =========================
# 2. Clean Column Names
# =========================
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace(".", "", regex=False)

print("Columns:", df.columns)

# =========================
# 3. Handle Missing Values
# =========================
df = df.dropna()

# =========================
# 4. Feature Engineering
# =========================
df["Price_per_sqft"] = df["Price"] / df["Area"]

# =========================
# 5. Define Target (LOG TRANSFORM)
# =========================
y = np.log(df["Price"])

# =========================
# 6. Select Features (REMOVE leakage)
# =========================
X = df.select_dtypes(include=[np.number]).drop(["Price", "Price_per_sqft"], axis=1)

# =========================
# 7. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 8. Train Model (XGBoost)
# =========================
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5
)

model.fit(X_train, y_train)

# =========================
# 9. Predictions
# =========================
y_pred = model.predict(X_test)

# Convert back from log scale
y_pred = np.exp(y_pred)
y_test = np.exp(y_test)

# =========================
# 10. Evaluation
# =========================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# =========================
# 11. Visualization
# =========================
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
# =========================
# 12. Feature Importance
# =========================
import pandas as pd
import matplotlib.pyplot as plt

importance = model.feature_importances_
features = X.columns

feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)

plt.figure()
feat_imp.head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# =========================
# 13. SHAP Explainability
# =========================
import shap

# Create explainer
explainer = shap.Explainer(model)

# Compute SHAP values
shap_values = explainer(X_test)

# =========================
# 1. Summary Plot (GLOBAL)
# =========================
shap.summary_plot(shap_values, X_test)

# =========================
# 2. Bar Plot (Feature Importance)
# =========================
shap.plots.bar(shap_values)
# =========================
# 3. Single Prediction Explanation
# =========================
shap.plots.waterfall(shap_values[0])