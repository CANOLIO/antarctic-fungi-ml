
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

N_GROUP_FOLDS = 5

DATA_FILE     = os.path.join("data", "processed", "dataset_features.csv")
MANIFEST_FILE = os.path.join("results", "models", "split_manifest.json")
FEAT_COLS_FILE = os.path.join("results", "models", "feature_columns.txt")


def find_best_threshold(probs_cold, y_true):
    best_t, best_f2 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 100):
        f2 = fbeta_score(1 - y_true, (probs_cold >= t).astype(int), beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    return best_t, best_f2


def find_threshold_via_oof_cv(X_tr, y_tr, groups_tr, n_splits=N_GROUP_FOLDS):
    """Umbral F2-óptimo de LogReg fijado SOLO con train (GroupKFold por
    organismo, predicciones out-of-fold). El test set no interviene aquí."""
    n_groups = len(np.unique(groups_tr))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        print("  ⚠️  Muy pocos organismos en train para CV de umbral — usando 0.5.")
        return 0.5

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(y_tr))
    oof_mask  = np.zeros(len(y_tr), dtype=bool)

    for tr_idx, val_idx in gkf.split(X_tr, y_tr, groups_tr):
        if len(np.unique(y_tr[val_idx])) < 2:
            continue
        sc = StandardScaler()
        X_tr_fold = sc.fit_transform(X_tr.iloc[tr_idx])
        X_val_fold = sc.transform(X_tr.iloc[val_idx])
        fold_lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=2)
        fold_lr.fit(X_tr_fold, y_tr[tr_idx])
        oof_probs[val_idx] = fold_lr.predict_proba(X_val_fold)[:, 0]
        oof_mask[val_idx] = True

    if oof_mask.sum() == 0:
        print("  ⚠️  Sin predicciones OOF válidas — usando 0.5.")
        return 0.5

    threshold, f2_oof = find_best_threshold(oof_probs[oof_mask], y_tr[oof_mask])
    print(f"  Umbral LogReg fijado vía CV out-of-fold en TRAIN "
          f"(n={oof_mask.sum():,}): {threshold:.4f}  (F2 OOF = {f2_oof:.4f})")
    return threshold


if not os.path.exists(MANIFEST_FILE):
    raise FileNotFoundError(
        f"No existe {MANIFEST_FILE}. Corre primero 05_train_model.py (v4) — "
        f"ese script genera el manifiesto de split por organismo que este "
        f"script reutiliza en vez de re-derivar su propio split."
    )

with open(MANIFEST_FILE) as f:
    manifest = json.load(f)

fcols = open(FEAT_COLS_FILE).read().strip().split('\n')
df = pd.read_csv(DATA_FILE)

train_ids = set(manifest['train_protein_ids'])
test_ids  = set(manifest['test_protein_ids'])
train_df  = df[df['Protein_ID'].isin(train_ids)]
test_df   = df[df['Protein_ID'].isin(test_ids)]

X_tr = train_df[fcols].astype(np.float32)
X_te = test_df[fcols].astype(np.float32)
y_tr = train_df['Thermal_Class'].values
y_te = test_df['Thermal_Class'].values
groups_tr = train_df['Organism_Source'].values

# Umbral propio para LogReg, fijado SOLO con train (CV out-of-fold, sin test)
print("Fijando umbral (CV out-of-fold en train, sin tocar test)...")
thresh_lr = find_threshold_via_oof_cv(X_tr, y_tr, groups_tr)

# Modelo final: se refita en TODO train, se evalúa UNA vez en test
scaler = StandardScaler()
lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=2)
lr.fit(scaler.fit_transform(X_tr), y_tr)
probs_lr = lr.predict_proba(scaler.transform(X_te))[:, 0]

cold_pred = (probs_lr >= thresh_lr).astype(int)   # 1 = predicho Cold
y_pred    = 1 - cold_pred                          # encoding Thermal_Class (0=cold,1=warm)
cold_true = 1 - y_te

print("\n=== Logistic Regression (umbral fijado en train, split por organismo) ===")
print(f"Umbral F2-óptimo (LogReg) : {thresh_lr:.4f}  (referencia: LightGBM usa {manifest['threshold']:.4f})")
print(f"F2-Score  : {fbeta_score(cold_true, cold_pred, beta=2):.4f}")
print(f"Precision : {precision_score(cold_true, cold_pred, zero_division=0):.2%}")
print(f"Recall    : {recall_score(cold_true, cold_pred, zero_division=0):.2%}")
print(f"Accuracy  : {accuracy_score(y_te, y_pred):.2%}")
print(f"\n(Comparar contra AUC LightGBM held-out por organismo: {manifest['auc_organism_holdout']:.4f})")