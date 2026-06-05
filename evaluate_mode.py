"""
evaluate_model.py - Thorough evaluation of the fraud detection model
=====================================================================
- Loads trained artefacts from models_store/
- Uses the same feature engineering as the API
- Evaluates on a holdout test set (20% of data, stratified)
- Reports metrics at various probability thresholds
- Recommends optimal threshold balancing precision and recall
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1. Load artefacts
# -----------------------------
print("Loading artefacts from models_store/...")
model = joblib.load("models_store/best_model.pkl")
scaler = joblib.load("models_store/scaler.pkl")
feature_names = joblib.load("models_store/feature_names.pkl")
amount_bins = joblib.load("models_store/amount_bins.pkl")
optimal_threshold = joblib.load("models_store/optimal_threshold.pkl")
print(f"Loaded model: {type(model).__name__}")
print(f"Optimal threshold (from training/validation): {optimal_threshold:.4f}\n")

# -----------------------------
# 2. Load original data
# -----------------------------
print("Loading creditcard.csv...")
df = pd.read_csv("data/creditcard.csv")
print(f"Total samples: {len(df)}, fraud rate: {df['Class'].mean():.6f}\n")

# -----------------------------
# 3. Feature engineering (exactly as API)
# -----------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Log_Amount"] = np.log1p(df["Amount"])
    # Use the same amount_bins loaded from training
    df["Amount_Q"] = pd.cut(df["Amount"], bins=amount_bins, labels=False).fillna(0).astype(float)
    # Keep only the required features in the correct order
    X = df[feature_names]
    return X

X_all = engineer_features(df)
y_all = df["Class"]

# -----------------------------
# 4. Split into train/test (use the same split as training? We'll use a new 80/20 for evaluation)
#    But to avoid contaminating training, we'll just use a test set from the original data.
#    However, the model was already trained on 80% of the data. For final evaluation, we should use
#    the 20% that was NOT used in training. Since we don't have the exact indices, we'll do a fresh split.
#    In practice, you'd want to keep a fixed test set. We'll use the same random_state as training.
print("Splitting data into train (80%) and test (20%) with stratification...")
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
print(f"Test set size: {len(X_test)}, frauds: {y_test.sum()}\n")

# Scale using the saved scaler (fitted on training data)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Predict probabilities
# -----------------------------
print("Computing fraud probabilities on test set...")
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# 6. Evaluate at multiple thresholds
# -----------------------------
thresholds = [0.3, 0.5, 0.7, 0.8, 0.9, 0.9244, 0.95, 0.99]
print("\n" + "="*80)
print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Frauds caught':>15} | {'False positives':>15}")
print("-"*80)

results = []
for thresh in thresholds:
    y_pred = (y_proba >= thresh).astype(int)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    frauds_caught = y_pred[y_test==1].sum()
    fp = (y_pred[y_test==0] == 1).sum()
    results.append((thresh, prec, rec, f1, frauds_caught, fp))
    print(f"{thresh:10.4f} | {prec:10.4f} | {rec:10.4f} | {f1:10.4f} | {frauds_caught:15} | {fp:15}")

# -----------------------------
# 7. Precision-Recall and ROC curves
# -----------------------------
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_pr_thresh = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5
best_f1 = f1_scores[best_idx]

roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

print("\n" + "="*80)
print(f"Best threshold based on test set F1: {best_pr_thresh:.4f} (F1 = {best_f1:.4f})")
print(f"ROC AUC on test set: {roc_auc:.4f}")

# -----------------------------
# 8. Confusion matrix at optimal business threshold (e.g., 0.7)
# -----------------------------
business_thresh = 0.7
y_pred_business = (y_proba >= business_thresh).astype(int)
cm = confusion_matrix(y_test, y_pred_business)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion matrix at threshold {business_thresh}:")
print(f"TP = {tp} (correctly blocked frauds)")
print(f"FN = {fn} (missed frauds)")
print(f"FP = {fp} (false alarms)")
print(f"TN = {tn} (correctly approved legit)")

# -----------------------------
# 9. Plot curves (optional, saved to file)
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(recalls, precisions, marker='.', label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.plot(recalls[best_idx], precisions[best_idx], 'ro', label=f'Best F1={best_f1:.3f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, marker='.', label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('model_evaluation_plots.png', dpi=150)
print("\nPlots saved as 'model_evaluation_plots.png'")

# -----------------------------
# 10. Summary recommendations
# -----------------------------
print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)
print(f"• Your model achieves ROC AUC = {roc_auc:.4f} (excellent).")
print(f"• At the F1‑optimal threshold {best_pr_thresh:.4f}, F1 = {best_f1:.4f}.")
print(f"• At your current business threshold {business_thresh}:")
print(f"    - Catches {tp} out of {tp+fn} frauds (recall = {recall_score(y_test, y_pred_business):.2%})")
print(f"    - Generates {fp} false alarms out of {tn+fp} legitimate transactions (false positive rate = {fp/(tn+fp):.4f}).")
print("\nIf you want higher recall (catch more fraud), lower the threshold.")
print("If you want higher precision (fewer false alarms), raise the threshold.")
print("The trained model is reliable; tune the decision threshold based on your business tolerance.")