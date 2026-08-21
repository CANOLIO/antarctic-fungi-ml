import os
import json
import argparse
import subprocess
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
N_GROUP_FOLDS       = 5
TEST_ORG_FRACTION   = 0.20

META_COLS  = ['Protein_ID', 'Organism_Source', 'Taxon_ID', 'T_opt_C',
              'Organism_Resolved', 'EC_Class', 'Thermal_Class']
TARGET_COL = 'Thermal_Class'
GROUP_COL  = 'Species_Group'


def get_species_name(org_str):
    s = str(org_str).replace('_', ' ').strip()
    parts = s.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return s


def get_domain(org_str):
    if any(g in str(org_str).lower() for g in FUNGI_GENERA):
        return 1  # Fungi
    return 0      # Bacteria


def load_and_balance(data_file):
    print("Cargando dataset de features...")
    df = pd.read_csv(data_file)
    df['Species_Group'] = df['Organism_Source'].apply(get_species_name)
    df['Is_Fungi'] = df['Organism_Source'].apply(get_domain)

    # 3-Tier Thermal Governance: Excluir psicrótrofos/ambiguos del entrenamiento y test binario principal
    psychrotroph_taxa = {'219572', '318456', '365044', '382245', '264483'}
    is_psychrotroph = (
        df['Taxon_ID'].astype(str).isin(psychrotroph_taxa) | 
        df['Organism_Source'].str.contains('antarctica|kishitanii|naphthalenivorans|rhodozyma', case=False, na=False) & (df[TARGET_COL] == 0)
    )
    
    # Shewanella oneidensis (Topt=30C) corregida a Warm (Clase 1)
    df.loc[df['Organism_Source'].str.contains('oneidensis', case=False, na=False), TARGET_COL] = 1

    df_primary = df[~is_psychrotroph].copy().reset_index(drop=True)
    df_psychrotrophs = df[is_psychrotroph].copy().reset_index(drop=True)
    
    # Guardar cohort psicrótrofo para análisis de sensibilidad aislado
    df_psychrotrophs.to_csv(os.path.join(RESULTS_DIR, "psychrotrophs_sensitivity_cohort.csv"), index=False)

    n_cold = (df_primary[TARGET_COL] == 0).sum()
    n_warm = (df_primary[TARGET_COL] == 1).sum()
    ratio  = n_warm / max(n_cold, 1)
    print(f"  [Primary Binary Cohort - Obligate Psychrophiles vs Mesophiles]")
    print(f"  ❄️  Cold (Obligate Psychrophiles): {n_cold:,}   🌱 Warm (Mesophiles): {n_warm:,}   Ratio: {ratio:.1f}x")
    print(f"  🦠 Bacteria: {(df_primary['Is_Fungi']==0).sum():,}  |  🍄 Fungi: {(df_primary['Is_Fungi']==1).sum():,}")
    print(f"  ⚠️  Quarantined Psychrotrophs (Sensitivity Benchmark): {len(df_psychrotrophs)} seqs")
    return df_primary, df_psychrotrophs


def calibrate_branch_oof(sub_df, feat_cols, model_type='bact', percentile=30):
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

    best_tau, best_f2 = 0.20, 0.0
    for t in np.linspace(0.08, 0.60, 53):
        f2 = fbeta_score(oof_y_val, (oof_p_val >= t).astype(int), beta=2.0, zero_division=0)
        if f2 > best_f2:
            best_f2, best_tau = f2, t

    print(f"  [{model_type.upper()}] Umbral calibrado en TRAIN OOF (n={len(oof_y_val)}): tau={best_tau:.4f} (F2 OOF={best_f2:.4f})")
    return best_tau


def train(data_file: str):
    print("\n" + "=" * 75)
    print("  PsychroScan v3.0 — Species-Disjoint Hierarchical Architecture")
    print("  Etapa 1 (Dominio) + Etapa 2A/2B (Desacoplada: 431 Bact / 434 Fungi)")
    print("=" * 75)

    df_primary, df_psychrotrophs = load_and_balance(data_file)
    all_feat_cols = [c for c in df_primary.columns if c not in META_COLS and c != 'Species_Group' and c != 'Is_Fungi']
    ptm_cols = ['N_Glyco_Density', 'N_Terminal_Hydrophobicity', 'Cys_Pair_Density']
    bact_feat_cols = [c for c in all_feat_cols if c not in ptm_cols]
    fungi_feat_cols = all_feat_cols

    # Split estrictamente Species-Disjoint
    species_df = df_primary[['Species_Group', TARGET_COL, 'Is_Fungi']].drop_duplicates('Species_Group').reset_index(drop=True)
    strat_key = species_df[TARGET_COL].astype(str) + "_" + species_df['Is_Fungi'].astype(str)

    train_species, test_species = train_test_split(
        species_df['Species_Group'],
        test_size=TEST_ORG_FRACTION,
        random_state=RANDOM_STATE,
        stratify=strat_key
    )

    train_df = df_primary[df_primary['Species_Group'].isin(train_species)].copy().reset_index(drop=True)
    test_df  = df_primary[df_primary['Species_Group'].isin(test_species)].copy().reset_index(drop=True)

    print(f"\n  Especies únicas en el dataset: {len(species_df)}")
    print(f"  Split estrictamente SPECIES-DISJOINT (GroupKFold-safe):")
    print(f"    Train: {len(train_df):,} seqs ({len(train_species)} especies) — Cold: {(train_df[TARGET_COL]==0).sum()}, Warm: {(train_df[TARGET_COL]==1).sum()}")
    print(f"    Test : {len(test_df):,} seqs ({len(test_species)} especies) — Cold: {(test_df[TARGET_COL]==0).sum()}, Warm: {(test_df[TARGET_COL]==1).sum()}")
    print(f"    Species Overlap Train ∩ Test: {len(set(train_species).intersection(set(test_species)))} (Zero Leakage Guaranteed)")

    # ── 1. Etapa 1: Clasificador de Dominio (Bacteria vs Fungi) ───────────────
    print("\n[Etapa 1] Entrenando Clasificador de Dominio (Bacteria vs Fungi)...")
    X_dom_tr = train_df[all_feat_cols].astype(np.float32)
    y_dom_tr = train_df['Is_Fungi'].values
    domain_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE))
    ])
    domain_pipe.fit(X_dom_tr, y_dom_tr)
    joblib.dump(domain_pipe, os.path.join(MODELS_DIR, "domain_classifier.pkl"))

    # ── 2. Etapa 2A: Rama Bacteriana (431 Features Biofísicos Puros) ──────────
    print("\n[Etapa 2A] Entrenando Rama Bacteriana (431 Biofísicos Puros)...")
    bact_tr = train_df[train_df['Is_Fungi'] == 0].copy()
    tau_b = calibrate_branch_oof(bact_tr, bact_feat_cols, model_type='bact', percentile=30)

    X_b_tr = bact_tr[bact_feat_cols].astype(np.float32)
    y_b_tr = (bact_tr[TARGET_COL] == 0).astype(int).values
    sel_b = SelectPercentile(mutual_info_classif, percentile=30)
    X_b_tr_s = sel_b.fit_transform(X_b_tr, y_b_tr)

    m_b_lgb = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=31, random_state=RANDOM_STATE, verbose=-1)
    m_b_rf  = RandomForestClassifier(n_estimators=250, max_depth=12, criterion='entropy', random_state=RANDOM_STATE, n_jobs=-1)
    m_b_et  = ExtraTreesClassifier(n_estimators=250, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)

    m_b_lgb.fit(X_b_tr_s, y_b_tr); m_b_rf.fit(X_b_tr_s, y_b_tr); m_b_et.fit(X_b_tr_s, y_b_tr)
    bact_branch = {'sel': sel_b, 'lgb': m_b_lgb, 'rf': m_b_rf, 'et': m_b_et}
    joblib.dump(bact_branch, os.path.join(MODELS_DIR, "branch_bacteria.pkl"))

    # ── 3. Etapa 2B: Rama Fúngica (434 Features con PTM Proxies) ──────────────
    print("\n[Etapa 2B] Entrenando Rama Fúngica (434 Features con PTM Proxies)...")
    fungi_tr = train_df[train_df['Is_Fungi'] == 1].copy()
    tau_f = calibrate_branch_oof(fungi_tr, fungi_feat_cols, model_type='fungi', percentile=20)

    X_f_tr = fungi_tr[fungi_feat_cols].astype(np.float32)
    y_f_tr = (fungi_tr[TARGET_COL] == 0).astype(int).values
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

    # ── 5. Evaluación End-to-End Canónica en Test Held-Out ────────────────────
    X_te = test_df[all_feat_cols].astype(np.float32)
    y_te_cold = (test_df[TARGET_COL] == 0).astype(int).values

    probs = hierarchical_model.predict_proba(X_te)
    probs_cold = probs[:, 0]
    preds_cold = (1 - hierarchical_model.predict(X_te))

    auc_global  = roc_auc_score(y_te_cold, probs_cold)
    acc_global  = accuracy_score(y_te_cold, preds_cold)
    prec_global = precision_score(y_te_cold, preds_cold, zero_division=0)
    rec_global  = recall_score(y_te_cold, preds_cold, zero_division=0)

    print("\n" + "=" * 75)
    print("  RESULTADO END-TO-END NO-ORÁCULO — Strictly SPECIES-DISJOINT")
    print(f"  ROC-AUC: {auc_global:.4f}  |  Accuracy: {acc_global*100:.2f}%  |  Precision: {prec_global*100:.2f}%  |  Recall: {rec_global*100:.2f}%")
    print("=" * 75)

    # ── 6. Desglose Estratificado por Dominio ─────────────────────────────────
    mask_b = (test_df['Is_Fungi'] == 0).values
    mask_f = (test_df['Is_Fungi'] == 1).values

    print("\n  ── Rendimiento Estratificado por Dominio (SPECIES-DISJOINT) ──")
    print(f"  🦠 BACTERIA (n={mask_b.sum()}): ROC-AUC = {roc_auc_score(y_te_cold[mask_b], probs_cold[mask_b]):.4f} | "
          f"Precision = {precision_score(y_te_cold[mask_b], preds_cold[mask_b])*100:.2f}% | "
          f"Recall = {recall_score(y_te_cold[mask_b], preds_cold[mask_b])*100:.2f}%")
    print(f"  🍄 FUNGI    (n={mask_f.sum()}): ROC-AUC = {roc_auc_score(y_te_cold[mask_f], probs_cold[mask_f]):.4f} | "
          f"Precision = {precision_score(y_te_cold[mask_f], preds_cold[mask_f])*100:.2f}% | "
          f"Recall = {recall_score(y_te_cold[mask_f], preds_cold[mask_f])*100:.2f}%")

    # ── 7. Generación del Registro Canónico (heldout_predictions.csv) ─────────
    commit_hash = "unknown"
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except Exception:
        pass

    pred_domains = domain_pipe.predict(X_te)
    canonical_df = pd.DataFrame({
        'Protein_ID':          test_df['Protein_ID'],
        'Species_Group':       test_df['Species_Group'],
        'Organism_Source':     test_df['Organism_Source'],
        'Domain_True':         test_df['Is_Fungi'].map({0: 'Bacteria', 1: 'Fungi'}),
        'Domain_Pred':         pd.Series(pred_domains).map({0: 'Bacteria', 1: 'Fungi'}),
        'True_Thermal_Class':  test_df[TARGET_COL],
        'P_Cold':              probs_cold,
        'Pred_Cold_Tau':       preds_cold,
        'Split_Version':       'v3.0.0_Species_Disjoint',
        'Feature_Set_Version': 'Decoupled_431Bact_434Fungi',
        'Model_Commit_Hash':   commit_hash
    })
    canonical_path = os.path.join(MODELS_DIR, "heldout_predictions.csv")
    canonical_df.to_csv(canonical_path, index=False)
    print(f"\n  📄 Registro Canónico Único guardado → {canonical_path} ({len(canonical_df)} filas)")

    # ── 8. Manifiesto y Metadata ──────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "threshold.txt"), 'w') as f:
        f.write(f"{tau_b},{tau_f}")
    with open(os.path.join(MODELS_DIR, "feature_columns.txt"), 'w') as f:
        f.write('\n'.join(all_feat_cols))

    manifest = {
        "data_file":            data_file,
        "random_state":         RANDOM_STATE,
        "split_unit":           "species",
        "split_version":        "v3.0.0_Species_Disjoint",
        "model_architecture":   "HierarchicalPsychroScan_SpeciesDisjoint",
        "tau_bacteria":         tau_b,
        "tau_fungi":            tau_f,
        "train_species":        sorted(train_species),
        "test_species":         sorted(test_species),
        "test_protein_ids":     list(test_df['Protein_ID'])
    }
    with open(os.path.join(MODELS_DIR, "split_manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  📄 Manifiesto jerárquico guardado → results/models/split_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PsychroScan v3.0 Species-Disjoint Trainer")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE, help="Ruta al CSV de features procesado")
    args = parser.parse_args()
    train(args.data_file)