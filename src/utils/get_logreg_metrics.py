import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (fbeta_score, accuracy_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df    = pd.read_csv('data/processed/dataset_features.csv')
fcols = open('results/models/feature_columns.txt').read().strip().split('\n')
X     = df[fcols].astype(np.float32)
y     = df['Thermal_Class'].values
df['strat_key'] = df['EC_Class'] + '_' + df['Thermal_Class'].astype(str)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=df['strat_key'], random_state=42)

scaler = StandardScaler()
lr     = LogisticRegression(max_iter=500, random_state=42, n_jobs=2)
lr.fit(scaler.fit_transform(X_tr), y_tr)
probs_lr = lr.predict_proba(scaler.transform(X_te))[:, 0]

thresh   = float(open('results/models/threshold.txt').read())
y_pred   = 1 - (probs_lr >= thresh).astype(int)
cold_true = 1 - y_te
cold_pred = 1 - y_pred

print("=== Logistic Regression (umbral LightGBM) ===")
print(f"F2-Score  : {fbeta_score(cold_true, cold_pred, beta=2):.4f}")
print(f"Precision : {precision_score(cold_true, cold_pred):.2%}")
print(f"Recall    : {recall_score(cold_true, cold_pred):.2%}")
print(f"Accuracy  : {accuracy_score(y_te, y_pred):.2%}")