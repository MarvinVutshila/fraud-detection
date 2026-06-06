#!/usr/bin/env python3
"""
live_stream_review_focused.py - Emphasises REVIEW decisions for human approval testing.
Pre‑computes which transactions lead to REVIEW (probability between 0.2 and 0.7)
and streams them along with a configurable mix of normal/fraud.
"""

import numpy as np
import pandas as pd
import requests
import time
import random
from datetime import datetime
import os

# ------------------------------------------------------------------
# Environment variables (do not hardcode secrets!)
# ------------------------------------------------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
USERNAME = os.getenv("STREAM_USERNAME", "analyst")
PASSWORD = os.getenv("STREAM_PASSWORD")
if not PASSWORD:
    raise ValueError("STREAM_PASSWORD environment variable not set. Please set it before running.")

# Configuration
REVIEW_RATIO = float(os.getenv("REVIEW_RATIO", "0.8"))
NORMAL_RATIO = float(os.getenv("NORMAL_RATIO", "0.1"))
FRAUD_RATIO = float(os.getenv("FRAUD_RATIO", "0.1"))
INTERVAL_RANGE = tuple(map(float, os.getenv("INTERVAL_RANGE", "1.0,3.0").split(',')))
CSV_PATH = os.getenv("CSV_PATH", "data/creditcard.csv")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "2000"))

# Pre‑computed pools
review_pool = []   # (row, probability)
normal_pool = []   # (row, probability)
fraud_pool = []    # (row, probability)

def get_token():
    resp = requests.post(f"{API_BASE}/auth/login", json={"username": USERNAME, "password": PASSWORD})
    if resp.status_code != 200:
        raise Exception(f"Authentication failed: {resp.text}")
    return resp.json()["access_token"]

def send_transaction(row, token, explain=False):
    """Send transaction to /predict. Returns (result, error)."""
    tx = row.drop("Class").to_dict()
    for k, v in tx.items():
        if isinstance(v, (np.float32, np.float64)):
            tx[k] = float(v)
    # Add unique transaction_id
    tx["transaction_id"] = f"REAL-{int(time.time())}-{random.randint(1000,9999)}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = f"{API_BASE}/predict?explain={'true' if explain else 'false'}"
    try:
        resp = requests.post(url, json=tx, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, str(e)

def build_pools():
    """Pre‑compute predictions without explanations (fast)."""
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    legit = df[df["Class"] == 0].copy()
    fraud = df[df["Class"] == 1].copy()
    
    legit_sample = legit.sample(n=min(MAX_BATCH_SIZE, len(legit)), random_state=42)
    fraud_sample = fraud.sample(n=min(MAX_BATCH_SIZE, len(fraud)), random_state=42)
    all_sample = pd.concat([legit_sample, fraud_sample], ignore_index=True)
    all_sample = all_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    token = get_token()
    print(f"Pre‑computing predictions for {len(all_sample)} transactions (without SHAP)...")
    for idx, row in all_sample.iterrows():
        if idx % 200 == 0:
            print(f"  Processed {idx}/{len(all_sample)}")
        result, err = send_transaction(row, token, explain=False)   # fast
        if err:
            print(f"  Error on tx {idx}: {err}")
            continue
        prob = result["fraud_probability"]
        dec = result["decision"]
        if dec == "REVIEW":
            review_pool.append((row, prob))
        elif prob < 0.2:
            normal_pool.append((row, prob))
        else:  # BLOCK or high prob
            fraud_pool.append((row, prob))
    
    print(f"Pool sizes: REVIEW={len(review_pool)}, NORMAL={len(normal_pool)}, FRAUD={len(fraud_pool)}")
    if len(review_pool) == 0:
        print("WARNING: No REVIEW transactions found. Lower your BLOCK_THRESHOLD in .env (e.g., 0.5) and restart API.")
    else:
        print(f"Example REVIEW probability: {review_pool[0][1]:.4f}")

def main():
    print("Building transaction pools (this may take a few minutes)...")
    build_pools()
    
    if len(review_pool) == 0:
        print("No REVIEW pool. Exiting.")
        return
    
    token = get_token()
    print("\nStreaming transactions (with SHAP only for REVIEWs). Press Ctrl+C to stop.\n")
    print(f"Stream composition: REVIEW={REVIEW_RATIO*100:.0f}%, NORMAL={NORMAL_RATIO*100:.0f}%, FRAUD={FRAUD_RATIO*100:.0f}%")
    print(f"Interval: {INTERVAL_RANGE[0]}-{INTERVAL_RANGE[1]} seconds\n")
    
    try:
        while True:
            r = random.random()
            if r < REVIEW_RATIO:
                row, prob = random.choice(review_pool)
                marker = "🚦 REVIEW"
                # For REVIEW, request SHAP explanations (slower but gives insights)
                result, err = send_transaction(row, token, explain=True)
            elif r < REVIEW_RATIO + NORMAL_RATIO:
                row, prob = random.choice(normal_pool)
                marker = "✓ NORMAL"
                result, err = send_transaction(row, token, explain=False)
            else:
                row, prob = random.choice(fraud_pool)
                marker = "💥 FRAUD"
                result, err = send_transaction(row, token, explain=False)
            
            if err:
                print(f"{marker} error: {err}")
                time.sleep(2)
                continue
            
            prob = result["fraud_probability"]
            dec = result["decision"]
            risk = result["risk_level"]
            expl = result.get("explanation")
            tx_id = row.get("transaction_id", f"REAL-{int(time.time())}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {marker} -> {tx_id[:20]} prob={prob:.4f} dec={dec} risk={risk}")
            if expl and "top_features" in expl:
                top = expl["top_features"][:3]
                contrib = expl["feature_contributions"][:3]
                print(f"    ➤ {', '.join([f'{f} ({c:+.4f})' for f,c in zip(top, contrib)])}")
            time.sleep(random.uniform(*INTERVAL_RANGE))
    except KeyboardInterrupt:
        print("\nStream stopped.")

if __name__ == "__main__":
    main()
