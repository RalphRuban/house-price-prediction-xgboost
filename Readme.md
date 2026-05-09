# 🏠 House Price Prediction using XGBoost

A Machine Learning project that predicts house prices using the XGBoost Regression algorithm and provides explainable AI insights using SHAP. The project also includes a Streamlit web application for real-time prediction and visualization.

---

# 📌 Features

- House price prediction using Machine Learning
- XGBoost Regression Model
- Streamlit Web Application
- SHAP Explainability
- Feature Importance Visualization
- Saved trained model using Joblib
- Real-time prediction interface

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- SHAP
- Matplotlib
- Joblib

---

# 📂 Project Structure

house-price-prediction-xgboost/

├── dataset/
│ └── Bangalore_House_Data.csv
│
├── models/
│ ├── model.pkl
│ └── columns.pkl
│
├── screenshots/
│ ├── app.png
│ ├── prediction.png
│ └── shap.png
│
├── train_model.py
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

---

# 📊 Dataset

Dataset used: Bangalore Housing Dataset

The dataset contains:
- Area
- Number of Bedrooms
- Amenities
- Property Features
- House Prices

Dataset Source:
https://www.kaggle.com/

---

# ⚙️ Installation

1️⃣ Clone Repository

```bash
git clone https://github.com/RalphRuban/house-price-prediction-xgboost.git
cd house-price-prediction-xgboost
```
2️⃣ Move into Project Folder
```bash
cd house-price-prediction-xgboost
```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
▶️ Run the Project

Train the Model
```bash
python train_model.py
```

Run Streamlit App
```bash
streamlit run app.py
```

🧠 Machine Learning Workflow

Load CSV Dataset
Data Preprocessing
Feature Engineering
Train XGBoost Model
Evaluate Model using RMSE
Save Model using Joblib
Deploy using Streamlit
Explain Predictions using SHAP

📈 Model Performance
Model Used: XGBoost Regressor
Evaluation Metric: RMSE
Dataset: Bangalore Housing Dataset

🔍 SHAP Explainability
SHAP (SHapley Additive Explanations) is integrated to explain predictions.

The system provides:
Feature contribution analysis
SHAP waterfall plots
Model transparency

📸 Screenshots
Streamlit Web App
![Dashboard.png](Dashboard.png)

Prediction Output
![Result_1.png](Result_1.png)

SHAP Explanation
![Figure_1.png](Figure_1.png)
![Figure_2.png](Figure_2.png)

🔮 Future Enhancements

Multi-city house price prediction
Cloud deployment
Deep learning integration
Real-time real estate API integration
Advanced feature engineering

👨‍💻 Author
Ralph Ruban M

⭐ Support
If you like this project, give it a star ⭐ on GitHub.