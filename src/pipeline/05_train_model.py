import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (classification_report, fbeta_score, f1_score,
                              roc_auc_score, confusion_matrix, precision_score,
                              recall_score, accuracy_score)
from sklearn.utils import resample

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.ensemble import HierarchicalPsychroScan, FUNGI_GENERA

# ─── RUTAS ────────────────────────────────────────────────────────────────────
DEFAULT_DATA_FILE = os.path.join("data", "processed", "dataset_features.csv")
MODELS_DIR  = os.path.join("results", "models")
RESULTS_DIR = os.path.join("results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── PARÁMETROS ───────────────────────────────────────────────────────────────
RANDOM_STATE        = 42
TOP15_MAX_PER_EC    = 5
N_GROUP_FOLDS       = 5
TEST_ORG_FRACTION   = 0.20

META_COLS  = ['Protein_ID', 'Organism_Source', 'Taxon_ID', 'T_opt_C',
              'Organism_Resolved', 'EC_Class', 'Thermal_Class']
TARGET_COL = 'Thermal_Class'
GROUP_COL  = 'Organism_Source'


def get_domain(org_str):
    if any(g in str(org_str).lower() for g in FUNGI_GENERA):
        return 1  # Fungi
    return 0      # Bacteria


def load_and_balance(data_file):
    print("Cargando dataset de features...")
    df = pd.read_csv(data_file)
    if GROUP_COL not in df.columns:
        raise ValueError(f"No existe la columna '{GROUP_COL}' en {data_file}.")
    df['Is_Fungi'] = df[GROUP_COL].apply(get_domain)
    n_cold = (df[TARGET_COL] == 0).sum()
    n_warm = (df[TARGET_COL] == 1).sum()
    ratio  = n_warm / max(n_cold, 1)
    print(f"  ❄️  Cold : {n_cold:,}   🌱 Warm : {n_warm:,}   Ratio: {ratio:.1f}x")
    print(f"  🦠 Bacteria: {(df['Is_Fungi']==0).sum():,}  |  🍄 Fungi: {(df['Is_Fungi']==1).sum():,}")
    return df


def split_by_organism(df):
    organisms = (df[[GROUP_COL, TARGET_COL, 'Is_Fungi']]
                 .drop_duplicates(subset=GROUP_COL)
                 .reset_index(drop=True))
    n_orgs = len(organisms)
    print(f"\n  Organismos únicos en el dataset: {n_orgs}")

    train_orgs, test_orgs = train_test_split(
        organisms[GROUP_COL],
        test_size=TEST_ORG_FRACTION,
        random_state=RANDOM_STATE,
        stratify=organisms[TARGET_COL].astype(str) + "_" + organisms['Is_Fungi'].astype(str),
    )

    train_mask = df[GROUP_COL].isin(set(train_orgs))
    test_mask  = df[GROUP_COL].isin(set(test_orgs))

    train_df = df[train_mask].reset_index(drop=True)
    test_df  = df[test_mask].reset_index(drop=True)

    overlap = set(train_df[GROUP_COL]).intersection(set(test_df[GROUP_COL]))
    assert len(overlap) == 0, f"LEAKAGE CRÍTICO: {len(overlap)} organismos compartidos!"

    feat_cols = [c for c in df.columns if c not in META_COLS and c != 'Is_Fungi']
    print(f"  Split por organismo (GroupKFold-safe):")
    print(f"    Train: {len(train_df):,} seqs ({len(train_orgs)} orgs) — "
          f"Cold: {(train_df[TARGET_COL]==0).sum():,}, Warm: {(train_df[TARGET_COL]==1).sum():,}")
    print(f"    Test : {len(test_df):,} seqs ({len(test_orgs)} orgs) — "
          f"Cold: {(test_df[TARGET_COL]==0).sum():,}, Warm: {(test_df[TARGET_COL]==1).sum():,}")
    print(f"    Features: {len(feat_cols)}")

    return train_df, test_df, feat_cols


def calibrate_branch_oof(train_df, is_fungi_val, feat_cols, percentile, model_type='bact'):
    sub_df = train_df[train_df['Is_Fungi'] == is_fungi_val].copy().reset_index(drop=True)
    X = sub_df[feat_cols].astype(np.float32)
    y = (sub_df[TARGET_COL] == 0).astype(int).values
    groups = sub_df[GROUP_COL].values

    gkf = GroupKFold(n_splits=min(N_GROUP_FOLDS, len(np.unique(groups))))
    oof_p = np.zeros(len(sub_df))
    oof_m = np.zeros(len(sub_df), dtype=bool)

    for tr_idx, val_idx in gkf.split(X, y, groups):
        if len(np.unique(y[val_idx])) < 2:
            continue
        sel = SelectPercentile(mutual_info_classif, percentile=percentile)
        X_tr_s = sel.fit_transform(X.iloc[tr_idx], y[tr_idx])
        X_va_s = sel.transform(X.iloc[val_idx])

        if model_type == 'bact':
            m_l = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=31, random_state=RANDOM_STATE, verbose=-1)
            m_r = RandomForestClassifier(n_estimators=250, max_depth=12, criterion='entropy', random_state=RANDOM_STATE, n_jobs=-1)
            m_e = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
            m_l.fit(X_tr_s, y[tr_idx]); m_r.fit(X_tr_s, y[tr_idx]); m_e.fit(X_tr_s, y[tr_idx])
            p = 0.50 * m_l.predict_proba(X_va_s)[:, 1] + 0.25 * m_r.predict_proba(X_va_s)[:, 1] + 0.25 * m_e.predict_proba(X_va_s)[:, 1]
        else:
            m_r = RandomForestClassifier(n_estimators=250, max_depth=8, criterion='entropy', random_state=RANDOM_STATE, n_jobs=-1)
            m_e = ExtraTreesClassifier(n_estimators=250, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
            m_l = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=15, random_state=RANDOM_STATE, verbose=-1)
            m_r.fit(X_tr_s, y[tr_idx]); m_e.fit(X_tr_s, y[tr_idx]); m_l.fit(X_tr_s, y[tr_idx])
            p = 0.40 * m_r.predict_proba(X_va_s)[:, 1] + 0.40 * m_e.predict_proba(X_va_s)[:, 1] + 0.20 * m_l.predict_proba(X_va_s)[:, 1]

        oof_p[val_idx] = p
        oof_m[val_idx] = True

    oof_y_val = y[oof_m]
    oof_p_val = oof_p[oof_m]

    best_tau, best_f1 = 0.25, 0.0
    for t in np.linspace(0.1, 0.8, 71):
        f1 = f1_score(oof_y_val, (oof_p_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, t

    print(f"  [{model_type.upper()}] Umbral calibrado en TRAIN OOF (n={len(oof_y_val)}): tau={best_tau:.4f} (F1 OOF={best_f1:.4f})")
    return best_tau


def train(data_file: str, run_legacy_comparison: bool):
    print("\n" + "=" * 75)
    print("  PsychroScan — Arquitectura Jerárquica Condicionada por Dominio (v6)")
    print("  Etapa 1 (Dominio) + Etapa 2A/2B (Especialización Térmica Desacoplada)")
    print("=" * 75 + "\n")

    df = load_and_balance(data_file)
    train_df, test_df, feat_cols = split_by_organism(df)

    # ── 1. Etapa 1: Clasificador de Dominio (solo ve train) ────────────────────
    print("\n[Etapa 1] Entrenando Clasificador de Dominio (Bacteria vs Fungi)...")
    domain_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, random_state=RANDOM_STATE))
    ])
    domain_pipe.fit(train_df[feat_cols].astype(np.float32), train_df['Is_Fungi'].values)
    joblib.dump(domain_pipe, os.path.join(MODELS_DIR, "domain_classifier.pkl"))

    # ── 2. Etapa 2A: Rama Bacteriana ──────────────────────────────────────────
    print("\n[Etapa 2A] Entrenando Rama Bacteriana (Feature Denoising + Ensamble)...")
    tau_b = calibrate_branch_oof(train_df, is_fungi_val=0, feat_cols=feat_cols, percentile=30, model_type='bact')

    bact_tr = train_df[train_df['Is_Fungi'] == 0]
    X_b_tr = bact_tr[feat_cols].astype(np.float32)
    y_b_tr = (bact_tr[TARGET_COL] == 0).astype(int)

    sel_b = SelectPercentile(mutual_info_classif, percentile=30)
    X_b_tr_s = sel_b.fit_transform(X_b_tr, y_b_tr)

    m_b_lgb = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=31, random_state=RANDOM_STATE, verbose=-1)
    m_b_rf  = RandomForestClassifier(n_estimators=250, max_depth=12, criterion='entropy', random_state=RANDOM_STATE, n_jobs=-1)
    m_b_et  = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)

    m_b_lgb.fit(X_b_tr_s, y_b_tr); m_b_rf.fit(X_b_tr_s, y_b_tr); m_b_et.fit(X_b_tr_s, y_b_tr)
    bact_branch = {'sel': sel_b, 'lgb': m_b_lgb, 'rf': m_b_rf, 'et': m_b_et}
    joblib.dump(bact_branch, os.path.join(MODELS_DIR, "branch_bacteria.pkl"))

    # ── 3. Etapa 2B: Rama Fúngica ─────────────────────────────────────────────
    print("\n[Etapa 2B] Entrenando Rama Fúngica (Feature Denoising + Ensamble)...")
    tau_f = calibrate_branch_oof(train_df, is_fungi_val=1, feat_cols=feat_cols, percentile=20, model_type='fungi')

    fungi_tr = train_df[train_df['Is_Fungi'] == 1]
    X_f_tr = fungi_tr[feat_cols].astype(np.float32)
    y_f_tr = (fungi_tr[TARGET_COL] == 0).astype(int)

    sel_f = SelectPercentile(mutual_info_classif, percentile=20)
    X_f_tr_s = sel_f.fit_transform(X_f_tr, y_f_tr)

    m_f_rf  = RandomForestClassifier(n_estimators=250, max_depth=8, criterion='entropy', random_state=RANDOM_STATE, n_jobs=-1)
    m_f_et  = ExtraTreesClassifier(n_estimators=250, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    m_f_lgb = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=15, random_state=RANDOM_STATE, verbose=-1)

    m_f_rf.fit(X_f_tr_s, y_f_tr); m_f_et.fit(X_f_tr_s, y_f_tr); m_f_lgb.fit(X_f_tr_s, y_f_tr)
    fungi_branch = {'sel': sel_f, 'rf': m_f_rf, 'et': m_f_et, 'lgb': m_f_lgb}
    joblib.dump(fungi_branch, os.path.join(MODELS_DIR, "branch_fungi.pkl"))

    # ── 4. Modelo Jerárquico Completo (End-to-End) ────────────────────────────
    hierarchical_model = HierarchicalPsychroScan(domain_pipe, bact_branch, fungi_branch, tau_b=tau_b, tau_f=tau_f)
    joblib.dump(hierarchical_model, os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))

    # ── 5. Evaluación End-to-End en Test Held-Out ─────────────────────────────
    X_te = test_df[feat_cols].astype(np.float32)
    y_te_cold = (test_df[TARGET_COL] == 0).astype(int).values

    probs = hierarchical_model.predict_proba(X_te)
    probs_cold = probs[:, 0]
    preds_cold = (1 - hierarchical_model.predict(X_te))  # predict devuelve 0=Cold, 1=Warm -> 1=Cold

    auc_global  = roc_auc_score(y_te_cold, probs_cold)
    acc_global  = accuracy_score(y_te_cold, preds_cold)
    prec_global = precision_score(y_te_cold, preds_cold, zero_division=0)
    rec_global  = recall_score(y_te_cold, preds_cold, zero_division=0)
    f1_global   = f1_score(y_te_cold, preds_cold, zero_division=0)

    print("\n" + "=" * 75)
    print("  RESULTADO END-TO-END NO-ORÁCULO — Held-out por ORGANISMO")
    print(f"  ROC-AUC: {auc_global:.4f}  |  Accuracy: {acc_global*100:.2f}%  |  Precision: {prec_global*100:.2f}%  |  Recall: {rec_global*100:.2f}%")
    print("=" * 75)
    print(classification_report(test_df[TARGET_COL].values, hierarchical_model.predict(X_te), target_names=['Cold (0)', 'Warm (1)']))

    # ── 6. Desglose Estratificado por Dominio ─────────────────────────────────
    mask_b = (test_df['Is_Fungi'] == 0).values
    mask_f = (test_df['Is_Fungi'] == 1).values

    print("\n  ── Rendimiento Estratificado por Dominio en Test Held-Out ──")
    print(f"  🦠 BACTERIA (n={mask_b.sum()}): ROC-AUC = {roc_auc_score(y_te_cold[mask_b], probs_cold[mask_b]):.4f} | "
          f"Accuracy = {accuracy_score(y_te_cold[mask_b], preds_cold[mask_b])*100:.2f}% | "
          f"Precision = {precision_score(y_te_cold[mask_b], preds_cold[mask_b])*100:.2f}% | "
          f"Recall = {recall_score(y_te_cold[mask_b], preds_cold[mask_b])*100:.2f}%")
    print(f"  🍄 FUNGI    (n={mask_f.sum()}): ROC-AUC = {roc_auc_score(y_te_cold[mask_f], probs_cold[mask_f]):.4f} | "
          f"Accuracy = {accuracy_score(y_te_cold[mask_f], preds_cold[mask_f])*100:.2f}% | "
          f"Precision = {precision_score(y_te_cold[mask_f], preds_cold[mask_f])*100:.2f}% | "
          f"Recall = {recall_score(y_te_cold[mask_f], preds_cold[mask_f])*100:.2f}%")

    # ── 7. Manifiesto y Metadata ──────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "threshold.txt"), 'w') as f:
        f.write(f"{tau_b},{tau_f}")
    with open(os.path.join(MODELS_DIR, "feature_columns.txt"), 'w') as f:
        f.write('\n'.join(feat_cols))

    manifest = {
        "data_file":            data_file,
        "random_state":         RANDOM_STATE,
        "split_unit":           "organism",
        "model_architecture":   "HierarchicalPsychroScan_TwoStage",
        "tau_bacteria":         tau_b,
        "tau_fungi":            tau_f,
        "threshold":            tau_b,  # backward compatibility default
        "train_organisms":      sorted(set(train_df['Organism_Source'])),
        "test_organisms":       sorted(set(test_df['Organism_Source'])),
        "train_protein_ids":    train_df['Protein_ID'].tolist(),
        "test_protein_ids":     test_df['Protein_ID'].tolist(),
        "auc_global_holdout":   auc_global,
        "auc_bacteria_holdout": roc_auc_score(y_te_cold[mask_b], probs_cold[mask_b]),
        "auc_fungi_holdout":    roc_auc_score(y_te_cold[mask_f], probs_cold[mask_f]),
    }
    manifest_path = os.path.join(MODELS_DIR, "split_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  📄 Manifiesto jerárquico guardado → {manifest_path}")

    # Top 15 diversificado
    top_15 = test_df[test_df[TARGET_COL] == 0].copy()
    top_15['Cold_Probability'] = probs_cold[test_df[TARGET_COL] == 0]
    top_15 = top_15.sort_values('Cold_Probability', ascending=False)
    selected = []
    ec_counts = {}
    for _, row in top_15.iterrows():
        ec = row['EC_Class']
        ec_counts[ec] = ec_counts.get(ec, 0)
        if ec_counts[ec] < TOP15_MAX_PER_EC:
            selected.append(row)
            ec_counts[ec] += 1
        if len(selected) >= 15:
            break
    top15_df = pd.DataFrame(selected)
    top15_df.to_csv(os.path.join(RESULTS_DIR, "top15_candidates_raw.csv"), index=False)
    print(f"  Top 15 guardado → results/top15_candidates_raw.csv\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default=DEFAULT_DATA_FILE, help='CSV de features')
    parser.add_argument('--skip-legacy-comparison', action='store_true', help='Omitir split legacy')
    args = parser.parse_args()
    train(args.data_file, run_legacy_comparison=not args.skip_legacy_comparison)