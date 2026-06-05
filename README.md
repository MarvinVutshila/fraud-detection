# 💳 Fraud Detection System (Machine Learning + Streamlit + FastAPI)

A production-style machine learning system for detecting fraudulent credit card transactions using XGBoost, Random Forest, Logistic Regression, and SHAP explainability.

---

## 🚀 Project Overview

This project detects fraudulent transactions using a trained ML pipeline built on the **Kaggle Credit Card Fraud Detection dataset**.

It includes:

- 🧠 Machine Learning models (XGBoost, Random Forest, Logistic Regression)
- ⚖️ SMOTE-based class balancing
- 📊 Performance evaluation (ROC-AUC, F1, Precision, Recall)
- 📉 Threshold optimization for fraud detection
- 🔍 SHAP explainability for model interpretation
- 🌐 Streamlit dashboard for real-time predictions
- ⚡ FastAPI backend for serving predictions

---

## 📁 Project Structure

```
fraud_detection/
├── data/
│   └── creditcard.csv
├── models/
│   ├── scaler.pkl
│   ├── xgboost.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── optimal_threshold.npy
│   └── feature_names.json
├── outputs/
│   ├── confusion_matrix plots
│   ├── roc_curves
│   └── SHAP explanations
├── api/
│   ├── fraud_api.py
│   └── sample_transaction.json
├── dashboard/
│   └── app.py
├── 01_eda.py
├── 02_preprocessing_and_modelling.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Important Architecture Note (VERY IMPORTANT)

The original dataset contains:

- V1 → V28 (PCA anonymised features)
- Time
- Amount

### ❌ Problem
In production, you cannot realistically provide V1–V28 features.

Using:
```
V1–V28 = 0.0
```

is **incorrect and unreliable**.

---

## ✅ Correct Production Design

The system must ONLY use features available at runtime:

### Input Features
- Amount (USD)
- Time (seconds)

### Derived Features (server-side)
- Hour = Time / 3600
- Log_Amount = log(Amount + 1)

### Final Feature Vector
```
[Hour, Log_Amount]
```

---

## 🔄 System Architecture

```
Streamlit UI
     ↓
FastAPI Backend
     ↓
Feature Engineering (Hour, Log_Amount)
     ↓
Trained ML Pipeline (XGBoost / RF)
     ↓
Fraud Probability Output
     ↓
Threshold Decision (Fraud / Legit)
```

---

## 📊 Model Performance

### XGBoost (Best Model)
- ROC-AUC: ~0.97
- F1 Score: ~0.60
- Precision (Fraud): ~0.46
- Recall (Fraud): ~0.87

### Random Forest
- ROC-AUC: ~0.98
- F1 Score: ~0.61

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 1. Train Models
```bash
python 02_preprocessing_and_modelling.py
```

### 2. Start FastAPI Backend
```bash
python api/fraud_api.py
```

### 3. Start Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🧪 Example Input

```
Amount: 500
Time: 43200 (12:00 PM)
```

Derived:
```
Hour = 12
Log_Amount = 6.216
```

---

## 🧠 Key Features

- Real-time fraud scoring
- Threshold tuning for business control
- SHAP explainability (why transaction is flagged)
- Lightweight model for laptop deployment
- Optimized memory pipeline

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Streamlit
- FastAPI
- SHAP
- Imbalanced-learn (SMOTE)

---

## 📌 Future Improvements

- Replace PCA dataset with real banking features
- Add transaction history features
- Deploy on cloud (AWS / Azure)
- Add real-time streaming (Kafka)
- Model monitoring (drift detection)

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.  
It should not be used directly in real financial systems without proper validation and compliance checks.

---

## 👨‍💻 Author

Marvin — Data Science & AI Enthusiast