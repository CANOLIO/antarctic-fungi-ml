import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
import joblib
import optuna
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (classification_report, fbeta_score, f1_score,
                              roc_auc_score, confusion_matrix, precision_score,
                              recall_score, accuracy_score)
from sklearn.utils import resample

# ─── RUTAS ────────────────────────────────────────────────────────────────────
DEFAULT_DATA_FILE = os.path.join("data", "processed", "dataset_features.csv")
MODELS_DIR  = os.path.join("results", "models")
RESULTS_DIR = os.path.join("results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── PARÁMETROS ───────────────────────────────────────────────────────────────
OPTUNA_TRIALS       = 30
MAX_WARM_MULTIPLIER = 4
RANDOM_STATE        = 42
TOP15_MAX_PER_EC    = 5
N_GROUP_FOLDS       = 5
TEST_ORG_FRACTION   = 0.20

META_COLS  = ['Protein_ID', 'Organism_Source', 'Taxon_ID', 'T_opt_C',
              'Organism_Resolved', 'EC_Class', 'Thermal_Class']
TARGET_COL = 'Thermal_Class'
GROUP_COL  = 'Organism_Source'


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.ensemble import PsychroScanEnsemble


def load_and_balance(data_file):
    print("Cargando dataset de features...")
    df = pd.read_csv(data_file)

    if GROUP_COL not in df.columns:
        raise ValueError(f"No existe la columna '{GROUP_COL}' en {data_file}.")

    n_cold = (df[TARGET_COL] == 0).sum()
    n_warm = (df[TARGET_COL] == 1).sum()
    ratio  = n_warm / max(n_cold, 1)
    print(f"  ❄️  Cold : {n_cold:,}   🌱 Warm : {n_warm:,}   Ratio: {ratio:.1f}x")

    if ratio > MAX_WARM_MULTIPLIER:
        target_warm = n_cold * MAX_WARM_MULTIPLIER
        print(f"  ⚖️  Undersampling Warm: {n_warm:,} → {target_warm:,}")
        df_warm_down = resample(df[df[TARGET_COL] == 1], n_samples=target_warm,
                                replace=False, random_state=RANDOM_STATE)
        df = pd.concat([df[df[TARGET_COL] == 0], df_warm_down]).sample(
            frac=1, random_state=RANDOM_STATE)
        print(f"  ✅ Nuevo balance → Cold: {(df[TARGET_COL]==0).sum():,}  "
              f"Warm: {(df[TARGET_COL]==1).sum():,}")
    else:
        print("  ✅ Ratio aceptable, sin undersampling.")

    return df


def split_by_organism(df):
    organisms = (df[[GROUP_COL, TARGET_COL]]
                 .drop_duplicates(subset=GROUP_COL)
                 .reset_index(drop=True))
    n_orgs = len(organisms)
    print(f"\n  Organismos únicos en el dataset: {n_orgs}")
    print(f"    Cold : {(organisms[TARGET_COL]==0).sum()}   "
          f"Warm : {(organisms[TARGET_COL]==1).sum()}")

    train_orgs, test_orgs = train_test_split(
        organisms[GROUP_COL],
        test_size=TEST_ORG_FRACTION,
        random_state=RANDOM_STATE,
        stratify=organisms[TARGET_COL],
    )

    train_mask = df[GROUP_COL].isin(set(train_orgs))
    test_mask  = df[GROUP_COL].isin(set(test_orgs))

    train_df = df[train_mask].reset_index(drop=True)
    test_df  = df[test_mask].reset_index(drop=True)

    overlap = set(train_df[GROUP_COL]).intersection(set(test_df[GROUP_COL]))
    assert len(overlap) == 0, f"LEAKAGE CRÍTICO: {len(overlap)} organismos compartidos!"

    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Split por organismo (GroupKFold-safe):")
    print(f"    Train: {len(train_df):,} seqs ({len(train_orgs)} orgs) — "
          f"Cold: {(train_df[TARGET_COL]==0).sum():,}, Warm: {(train_df[TARGET_COL]==1).sum():,}")
    print(f"    Test : {len(test_df):,} seqs ({len(test_orgs)} orgs) — "
          f"Cold: {(test_df[TARGET_COL]==0).sum():,}, Warm: {(test_df[TARGET_COL]==1).sum():,}")
    print(f"    Features: {len(feat_cols)}")

    X_tr = train_df[feat_cols].astype(np.float32)
    y_tr = train_df[TARGET_COL].values
    groups_tr = train_df[GROUP_COL].values

    X_te = test_df[feat_cols].astype(np.float32)
    y_te = test_df[TARGET_COL].values

    meta_tr = train_df[META_COLS]
    meta_te = test_df[META_COLS]

    return X_tr, X_te, y_tr, y_te, groups_tr, meta_tr, meta_te, feat_cols


def legacy_protein_level_split(df, feat_cols):
    X = df[feat_cols].astype(np.float32)
    y = df[TARGET_COL].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_ORG_FRACTION, random_state=RANDOM_STATE, stratify=y
    )
    return X_tr, X_te, y_tr, y_te


def objective(trial, X_tr, y_tr, groups_tr):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': 2,
    }
    gkf = GroupKFold(n_splits=N_GROUP_FOLDS)
    fold_aucs = []

    # Map target: 1 = Cold for classifier
    y_binary = (y_tr == 0).astype(int)

    for tr_idx, val_idx in gkf.split(X_tr, y_binary, groups_tr):
        if len(np.unique(y_binary[val_idx])) < 2:
            continue
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr.iloc[tr_idx], y_binary[tr_idx])
        probs = model.predict_proba(X_tr.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y_binary[val_idx], probs)
        fold_aucs.append(auc)

    return float(np.mean(fold_aucs)) if fold_aucs else 0.0


def find_threshold_via_oof_cv(X_tr, y_tr, groups_tr, lgb_params):
    """
    Fija el umbral óptimo de decisión evaluando predicciones OOF (Out-Of-Fold)
    estrictamente dentro de Train mediante GroupKFold.
    """
    gkf = GroupKFold(n_splits=N_GROUP_FOLDS)
    oof_probs = np.zeros(len(y_tr))
    oof_mask  = np.zeros(len(y_tr), dtype=bool)

    y_binary = (y_tr == 0).astype(int)

    for tr_idx, val_idx in gkf.split(X_tr, y_binary, groups_tr):
        if len(np.unique(y_binary[val_idx])) < 2:
            continue
        sel = SelectPercentile(mutual_info_classif, percentile=30)
        X_tr_f = sel.fit_transform(X_tr.iloc[tr_idx], y_binary[tr_idx])
        X_va_f = sel.transform(X_tr.iloc[val_idx])

        clf_lgb = lgb.LGBMClassifier(**lgb_params)
        clf_rf  = RandomForestClassifier(n_estimators=250, max_depth=12, criterion='entropy', random_state=RANDOM_STATE, n_jobs=-1)
        clf_et  = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)

        clf_lgb.fit(X_tr_f, y_binary[tr_idx])
        clf_rf.fit(X_tr_f, y_binary[tr_idx])
        clf_et.fit(X_tr_f, y_binary[tr_idx])

        p_l = clf_lgb.predict_proba(X_va_f)[:, 1]
        p_r = clf_rf.predict_proba(X_va_f)[:, 1]
        p_e = clf_et.predict_proba(X_va_f)[:, 1]

        oof_probs[val_idx] = 0.50 * p_l + 0.25 * p_r + 0.25 * p_e
        oof_mask[val_idx] = True

    oof_y = y_binary[oof_mask]
    oof_p = oof_probs[oof_mask]

    # Optimizar umbral por F1-Score equilibrado en OOF
    best_t, best_f1 = 0.25, 0.0
    for t in np.linspace(0.1, 0.8, 71):
        f1 = f1_score(oof_y, (oof_p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"  Umbral calibrado vía OOF CV en TRAIN (n={len(oof_y):,} seqs): "
          f"tau={best_t:.4f} (OOF F1={best_f1:.4f})")
    return best_t


def build_diverse_top15(meta_te, probs_cold, y_te, max_per_ec=TOP15_MAX_PER_EC):
    results = meta_te.copy()
    results['Cold_Probability'] = probs_cold
    cold_only = results[results['Thermal_Class'] == 0].sort_values(
        'Cold_Probability', ascending=False)
    selected, ec_counts = [], {}
    for _, row in cold_only.iterrows():
        ec = row['EC_Class']
        ec_counts[ec] = ec_counts.get(ec, 0)
        if ec_counts[ec] < max_per_ec:
            selected.append(row)
            ec_counts[ec] += 1
        if len(selected) >= 15:
            break
    return pd.DataFrame(selected)


def train(data_file: str, run_legacy_comparison: bool):
    print("\n" + "=" * 70)
    print("  PsychroScan — Entrenamiento Multimodelo Ensamble (v5)")
    print("  Feature Denoising + Disjoint GroupKFold Split")
    print("=" * 70 + "\n")

    df = load_and_balance(data_file)
    (X_tr, X_te, y_tr, y_te, groups_tr,
     meta_tr, meta_te, feat_cols) = split_by_organism(df)

    y_tr_binary = (y_tr == 0).astype(int)
    y_te_binary = (y_te == 0).astype(int)

    # 1. Optimizar LightGBM con Optuna
    print(f"\n🔍 Optimizando {OPTUNA_TRIALS} trials (Optuna ROC-AUC, GroupKFold interno x{N_GROUP_FOLDS})...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, groups_tr),
        n_trials=OPTUNA_TRIALS, show_progress_bar=True,
    )

    print(f"\n✅ Mejor ROC-AUC en CV interna de Train: {study.best_value:.4f}")
    best_lgb_params = {**study.best_params, 'n_jobs': 2, 'random_state': RANDOM_STATE, 'verbose': -1}

    # 2. Calibración de umbral OOF en Train
    print(f"\n🎯 Fijando umbral de decisión (OOF CV en Train)...")
    threshold = find_threshold_via_oof_cv(X_tr, y_tr, groups_tr, best_lgb_params)

    # 3. Entrenar Ensamble Final sobre TODO Train con Feature Selection
    print("\nEntrenando Ensamble Final (LightGBM + Random Forest + ExtraTrees)...")
    selector = SelectPercentile(mutual_info_classif, percentile=30)
    X_tr_sel = selector.fit_transform(X_tr, y_tr_binary)
    X_te_sel = selector.transform(X_te)

    m_lgb = lgb.LGBMClassifier(**best_lgb_params)
    m_rf  = RandomForestClassifier(n_estimators=300, max_depth=12, criterion='entropy', random_state=RANDOM_STATE, n_jobs=-1)
    m_et  = ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)

    m_lgb.fit(X_tr_sel, y_tr_binary)
    m_rf.fit(X_tr_sel, y_tr_binary)
    m_et.fit(X_tr_sel, y_tr_binary)

    ensemble = PsychroScanEnsemble(m_lgb, m_rf, m_et, selector=selector)

    # 4. Evaluación en Test Held-Out
    probs = ensemble.predict_proba(X_te)
    probs_cold = probs[:, 0]  # P(Cold)
    auc_score  = roc_auc_score(y_te_binary, probs_cold)

    cold_pred_te = (probs_cold >= threshold).astype(int)
    y_pred = 1 - cold_pred_te

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te_binary, cold_pred_te, zero_division=0)
    rec  = recall_score(y_te_binary, cold_pred_te, zero_division=0)
    f1   = f1_score(y_te_binary, cold_pred_te, zero_division=0)

    print("\n" + "=" * 70)
    print(f"  RESULTADO PRINCIPAL — Held-out por ORGANISMO (nunca visto en train)")
    print(f"  ROC-AUC: {auc_score:.4f}  |  Accuracy: {acc*100:.1f}%  |  Precision: {prec*100:.1f}%  |  Recall: {rec*100:.1f}%")
    print(f"  Umbral calibrado en Train (OOF F1): {threshold:.4f}")
    print("=" * 70)
    print(classification_report(y_te, y_pred, target_names=['Cold (0)', 'Warm (1)']))
    cm = confusion_matrix(y_te, y_pred)
    print(f"  Matriz de confusión:")
    print(f"              Pred Cold  Pred Warm")
    print(f"  Real Cold :   {cm[0,0]:>6}     {cm[0,1]:>6}")
    print(f"  Real Warm :   {cm[1,0]:>6}     {cm[1,1]:>6}")

    # 5. Comparación con Legacy
    legacy_auc = None
    if run_legacy_comparison:
        print("\n" + "-" * 70)
        print("  COMPARACIÓN — split legacy a nivel de proteína")
        print("-" * 70)
        X_tr_l, X_te_l, y_tr_l, y_te_l = legacy_protein_level_split(df, feat_cols)
        y_tr_l_bin = (y_tr_l == 0).astype(int)
        y_te_l_bin = (y_te_l == 0).astype(int)

        sel_l = SelectPercentile(mutual_info_classif, percentile=30)
        X_tr_l_sel = sel_l.fit_transform(X_tr_l, y_tr_l_bin)
        X_te_l_sel = sel_l.transform(X_te_l)

        m_lgb_l = lgb.LGBMClassifier(**best_lgb_params)
        m_rf_l  = RandomForestClassifier(n_estimators=300, max_depth=12, criterion='entropy', random_state=RANDOM_STATE, n_jobs=-1)
        m_et_l  = ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)

        m_lgb_l.fit(X_tr_l_sel, y_tr_l_bin)
        m_rf_l.fit(X_tr_l_sel, y_tr_l_bin)
        m_et_l.fit(X_tr_l_sel, y_tr_l_bin)

        ens_l = PsychroScanEnsemble(m_lgb_l, m_rf_l, m_et_l, selector=sel_l)
        legacy_probs = ens_l.predict_proba(X_te_l)[:, 0]
        legacy_auc = roc_auc_score(y_te_l_bin, legacy_probs)
        delta = legacy_auc - auc_score
        print(f"  AUC (split proteína, organismos mezclados) : {legacy_auc:.4f}")
        print(f"  AUC (split organismo, held-out real)       : {auc_score:.4f}")
        print(f"  Δ atribuible a leakage por organismo       : {delta:+.4f}")

    # 6. Guardar Modelo y Manifiesto
    joblib.dump(ensemble, os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    with open(os.path.join(MODELS_DIR, "threshold.txt"), 'w') as f:
        f.write(str(threshold))
    with open(os.path.join(MODELS_DIR, "feature_columns.txt"), 'w') as f:
        f.write('\n'.join(feat_cols))

    manifest = {
        "data_file":            data_file,
        "random_state":         RANDOM_STATE,
        "split_unit":           "organism",
        "train_organisms":      sorted(set(meta_tr['Organism_Source'])),
        "test_organisms":       sorted(set(meta_te['Organism_Source'])),
        "train_protein_ids":    meta_tr['Protein_ID'].tolist(),
        "test_protein_ids":     meta_te['Protein_ID'].tolist(),
        "auc_organism_holdout": auc_score,
        "auc_legacy_protein_split": legacy_auc,
        "threshold":            threshold,
        "threshold_method":     "oof_cv_train_only",
    }
    manifest_path = os.path.join(MODELS_DIR, "split_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  📄 Manifiesto de split guardado → {manifest_path}")

    # 7. Top 15 Diversificado
    top_15 = build_diverse_top15(meta_te, probs_cold, y_te)
    print("\n" + "=" * 70)
    print(f"  TOP 15 DIVERSIFICADO (max {TOP15_MAX_PER_EC} por EC_Class, organismos held-out)")
    print("=" * 70)
    for _, row in top_15.iterrows():
        print(f"  {row['Protein_ID'][:33]:<34} {row['EC_Class']:<22} "
              f"{row['Organism_Source']:<28} {row['Cold_Probability']*100:.2f}%")

    top_15.to_csv(os.path.join(RESULTS_DIR, "top15_candidates_raw.csv"), index=False)
    print(f"\n  Top 15 guardado → results/top15_candidates_raw.csv\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default=DEFAULT_DATA_FILE, help='CSV de features')
    parser.add_argument('--skip-legacy-comparison', action='store_true', help='Omitir split legacy')
    args = parser.parse_args()
    train(args.data_file, run_legacy_comparison=not args.skip_legacy_comparison)