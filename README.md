markdown
# 🛡️ Fraud Detection System – Production ML + API + Dashboard

End‑to‑end fraud detection using **XGBoost** (trained on the Kaggle Credit Card Fraud dataset), served via **FastAPI**, with a **Streamlit** dashboard and **PostgreSQL** for history. Fully containerised and deployable for free on Render + Neon + Streamlit Cloud.

## 🎯 Real System Architecture

This is exactly what you built and tested:

```mermaid
flowchart LR
    User[User / CSV] --> Dashboard[Streamlit Dashboard]
    Dashboard --> API[FastAPI Backend]
    API --> DB[(PostgreSQL / Neon)]
    API --> Model[Trained XGBoost Model]
    Model --> SHAP[SHAP Explanations]
    DB --> Dashboard
Important – the model uses the full 30 features of the original dataset:

Time (seconds) → derived Hour, Hour_sin, Hour_cos

Amount → Log_Amount, Amount_Q (quantile bin)

V1 … V28 (PCA‑anonymised features)

→ Your API requires V1‑V28. They are not set to zero. The dashboard sends them from your CSV.

📁 Project Structure (real)
text
fraud_detection/
├── data/
│   └── creditcard.csv               # original dataset (not needed for deployment)
├── models_store/                    # trained artefacts
│   ├── best_model.pkl               # XGBoost model
│   ├── scaler.pkl
│   ├── amount_bins.pkl
│   ├── feature_names.pkl
│   └── optimal_threshold.pkl
├── fraud_detection/                 # main package
│   ├── api/routes.py                # FastAPI endpoints
│   ├── core/config.py
│   ├── database/postgres_db.py
│   ├── models/model_loader.py
│   ├── services/                    # prediction, decision, storage
│   └── utils/                       # feature engineering, SHAP
├── dashboard/app.py                 # Streamlit frontend
├── main.py                          # FastAPI entry point
├── train.py                         # training with SMOTE + XGBoost
├── requirements.txt
├── runtime.txt                      # python-3.11.0
└── README.md
🧠 Model Performance (on test set)
Metric	Value
ROC‑AUC	0.9816
F1 (optimal threshold 0.9244)	0.8677
Recall at threshold 0.7	85.7%
False positive rate (0.7)	0.07%
→ Model is production‑ready for low‑false‑alarm fraud detection.

🚀 Free Deployment (no credit card)
We use three free services:

Component	Platform	Free limits
API	Render	750 hours/month, spins down after 15 min idle
Database	Neon	0.5 GB storage, serverless, scales to zero
Dashboard	Streamlit Cloud	Unlimited public apps
Step 1 – Neon Database
Sign up at neon.tech (GitHub login).

Create a project → copy the connection string (looks like postgresql://…).

Keep it – you'll need it for Render.

Step 2 – Deploy FastAPI on Render
Push your code to GitHub (e.g., MarvinVutshila/fraud-detection).

Log into Render → New Web Service → connect repo.

Use these settings:

Field	Value
Name	fraud-detection-api
Environment	Python
Build Command	python -m pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
Start Command	uvicorn main:app --host 0.0.0.0 --port 10000
Instance Type	Free
Environment Variables (secrets):

Key	Value
DATABASE_URL	(the Neon connection string)
API_KEY	changeme (or a strong random string)
APPROVE_THRESHOLD	0.2
BLOCK_THRESHOLD	0.7
Click Create Web Service.

✅ After ~3‑5 minutes, your API is live at https://fraud-detection-api.onrender.com.
Test: https://fraud-detection-api.onrender.com/health → {"status":"ok"}

Step 3 – Deploy Streamlit Dashboard on Streamlit Cloud
Push dashboard/app.py to the same GitHub repo.

Go to Streamlit Cloud → New app → select repo, branch, and dashboard/app.py.

In Advanced settings → Secrets, add:

text
API_URL = "https://fraud-detection-api.onrender.com"
API_KEY = "changeme"
Click Deploy.

Your dashboard will be available at https://your-app-name.streamlit.app.

🧪 Testing the Live System
Upload a CSV with columns: Time, Amount, V1, V2, …, V28 (exactly as the original dataset).

The dashboard sends a batch request to the API.

The API returns fraud probability, decision (APPROVE / BLOCK), risk level, and SHAP explanations (if installed).

Results are stored in Neon and displayed in the dashboard's history.

🔁 Updating
API: push changes → Render auto‑redeploys.

Dashboard: push changes → Streamlit Cloud auto‑redeploys.

Database: manage via Neon console.

⚠️ Important Notes for Production
Cold starts: Render spins down after 15 minutes – first request may take 30‑50 seconds.

Never use Render's free PostgreSQL – it expires after 30 days. Always use Neon or Supabase.

The model expects all V1‑V28 features – your CSV must include them (they are PCA components from the original dataset).
In a real‑world deployment, you would replace V1‑V28 with actual banking features; this is a proof‑of‑concept using the Kaggle dataset.

🛠 Troubleshooting
Problem	Solution
Build fails on Render	Ensure runtime.txt contains python-3.11.0 and Build Command includes pip install --upgrade pip setuptools wheel.
ModuleNotFoundError	Check that all dependencies are in requirements.txt (especially psycopg2-binary).
Database connection refused	Verify DATABASE_URL secret is correct; Neon database may be paused – resume it.
API returns 500 on /predict	Look at Render logs – likely a missing model file or column name mismatch.
📜 License
Educational / research use only. Not for live financial systems without proper validation.

👨‍💻 Author
Marvin – Data Science & AI Engineer




## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```