#!/usr/bin/env python3
"""
train.py - Fraud Detection Model Training with Imbalance Handling
=================================================================
- Loads creditcard.csv
- Engineers features (matching API)
- Applies SMOTE + class_weight
- Trains XGBoost (or RandomForest) with hyperparameter tuning
- Finds optimal threshold (max F1 on validation)
- Saves all artefacts for API
- Includes progress bars and configurable pauses.
"""

import os
import sys
import time
import gc
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

# Optional XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed, falling back to RandomForest.")

if not XGB_AVAILABLE:
    from sklearn.ensemble import RandomForestClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "data/creditcard.csv"
MODELS_DIR = "models_store"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_SAMPLING_STRATEGY = 0.3      # fraud proportion becomes 30% after SMOTE
USE_XGBOOST = XGB_AVAILABLE

# Pause durations (seconds)
PAUSE_AFTER_LOAD = 15
PAUSE_AFTER_SCALE = 30
PAUSE_AFTER_SMOTE = 30
PAUSE_AFTER_TRAIN = 20
PAUSE_AFTER_THRESHOLD = 10

# Hyperparameters
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 5,
    'eval_metric': 'auc',
    'use_label_encoder': False,
    'random_state': RANDOM_STATE,
    'verbosity': 0
}

RF_PARAMS = {
    'n_estimators': 150,
    'max_depth': 12,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}

# ============================================================
# UTILITIES
# ============================================================
def pause(seconds, message="Cooling down..."):
    if seconds <= 0:
        return
    logger.info(f"{message} Waiting {seconds} seconds...")
    for _ in tqdm(range(seconds), desc="Pause", unit="s"):
        time.sleep(1)
    gc.collect()

def engineer_features(df):
    logger.info("Engineering features...")
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Log_Amount"] = np.log1p(df["Amount"])
    
    amount_bins = np.quantile(df["Amount"], np.linspace(0, 1, 11))
    df["Amount_Q"] = pd.cut(df["Amount"], bins=amount_bins, labels=False).fillna(0).astype(float)
    
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Hour", "Hour_sin", "Hour_cos", "Log_Amount", "Amount_Q"]
    return df[feature_cols], amount_bins, feature_cols

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    start_time = time.time()
    logger.info("=== FRAUD DETECTION MODEL TRAINING ===")
    
    # 1. Load data
    logger.info(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Shape: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    fraud_rate = df['Class'].mean()
    logger.info(f"Fraud rate: {fraud_rate:.6f} ({df['Class'].sum()} frauds out of {len(df)})")
    pause(PAUSE_AFTER_LOAD, "After data loading")
    
    # 2. Feature engineering
    X, amount_bins, feature_cols = engineer_features(df)
    y = df['Class']
    logger.info(f"Features created: {len(feature_cols)} columns")
    del df
    gc.collect()
    
    # 3. Train/validation split
    logger.info("Splitting data (80% train, 20% validation) - stratified...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train size: {len(X_train)} (fraud rate: {y_train.mean():.5f})")
    logger.info(f"Val size:   {len(X_val)} (fraud rate: {y_val.mean():.5f})")
    
    # 4. Scaling
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    pause(PAUSE_AFTER_SCALE, "After scaling")
    
    # 5. SMOTE
    logger.info(f"Applying SMOTE (sampling_strategy={SMOTE_SAMPLING_STRATEGY})...")
    smote = SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    logger.info(f"After SMOTE: {len(X_train_res)} samples, fraud rate: {y_train_res.mean():.3f}")
    pause(PAUSE_AFTER_SMOTE, "After SMOTE (system rest)")
    
    # 6. Model training
    model = None
    if USE_XGBOOST:
        logger.info("Training XGBoost classifier...")
        model = xgb.XGBClassifier(**XGB_PARAMS)
        try:
            # Try modern syntax with early_stopping_rounds
            model.fit(
                X_train_res, y_train_res,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        except TypeError:
            # Older XGBoost version: early_stopping_rounds not accepted
            logger.warning("XGBoost version does not support early_stopping_rounds; training without early stopping.")
            model.fit(X_train_res, y_train_res, eval_set=[(X_val_scaled, y_val)], verbose=False)
    else:
        logger.info("Training RandomForest classifier...")
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train_res, y_train_res)
    
    logger.info("Model training finished.")
    pause(PAUSE_AFTER_TRAIN, "After training (system rest)")
    
    # 7. Optimal threshold
    logger.info("Computing validation probabilities and optimal threshold...")
    probs_val = model.predict_proba(X_val_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    logger.info(f"Optimal threshold (max F1): {optimal_threshold:.4f}")
    
    preds_val = (probs_val >= optimal_threshold).astype(int)
    val_f1 = f1_score(y_val, preds_val)
    val_auc = roc_auc_score(y_val, probs_val)
    logger.info(f"Validation F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
    pause(PAUSE_AFTER_THRESHOLD, "After threshold calculation")
    
    # 8. Save artefacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Saving artefacts to {MODELS_DIR}/")
    
    joblib.dump(model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(amount_bins, f"{MODELS_DIR}/amount_bins.pkl")
    joblib.dump(feature_cols, f"{MODELS_DIR}/feature_names.pkl")
    joblib.dump(optimal_threshold, f"{MODELS_DIR}/optimal_threshold.pkl")
    
    metadata = {
        "model_type": type(model).__name__,
        "fraud_rate_raw": fraud_rate,
        "fraud_rate_after_smote": float(y_train_res.mean()),
        "optimal_threshold": float(optimal_threshold),
        "validation_f1": val_f1,
        "validation_auc": val_auc,
        "feature_count": len(feature_cols),
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(metadata, f"{MODELS_DIR}/metadata.pkl")
    
    logger.info("Files saved:")
    for f in sorted(os.listdir(MODELS_DIR)):
        logger.info(f"  - {f}")
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed/60:.1f} minutes.")

if __name__ == "__main__":
    main()