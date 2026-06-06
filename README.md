🛡️ FRAUD DETECTION SYSTEM – PRODUCTION ML + API + DASHBOARD

End-to-end fraud detection using XGBoost (trained on the Kaggle Credit Card Fraud dataset), served via a modular FastAPI backend, with an HTML/JavaScript dashboard and PostgreSQL for transaction history.

Project Structure

fraud_detection/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── main.py
├── train.py
├── evaluate_model.py
├── live_stream_review_focused.py
│
├── tests/
│   ├── __init__.py
│   └── test_connection.py
│
├── frontend/
│   ├── index.html
│   └── Marvin.jpg
│
├── models_store/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── amount_bins.pkl
│   ├── feature_names.pkl
│   └── optimal_threshold.pkl
│
└── fraud_detection/
    ├── api/
    │   ├── dependencies.py
    │   ├── auth.py
    │   └── routes/
    │       ├── health.py
    │       ├── model.py
    │       ├── predictions.py
    │       ├── transactions.py
    │       ├── history.py
    │       ├── ingest.py
    │       └── auth.py
    │
    ├── application/
    │   └── services/
    │       ├── prediction_service.py
    │       └── decision_service.py
    │
    ├── infrastructure/
    │   ├── database/
    │   └── repositories/
    │
    ├── ml/
    │   ├── feature_engineering.py
    │   └── inference/
    │       ├── model_loader.py
    │       └── explainability.py
    │
    ├── schemas/
    ├── core/
    └── utils/

Model Performance

• ROC-AUC: 0.9816
• F1 Score: 0.8677
• Recall: 85.7%
• False Positive Rate: 0.07%

Deployment Stack

• FastAPI API → Render
• PostgreSQL → Neon
• Dashboard → Netlify / Vercel

Author

Marvin
Data Science & AI Engineer
