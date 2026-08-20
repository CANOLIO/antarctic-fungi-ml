import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import optuna
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (classification_report, fbeta_score,
                              roc_auc_score, confusion_matrix)
from sklearn.utils import resample

# ─── RUTAS ────────────────────────────────────────────────────────────────────
# ÚNICO punto de verdad para el archivo de datos (antes: 05 usaba _nr90.csv
# mientras 08/09/11/blast_benchmark/Brenda_validation usaban dataset_features.csv
# — esa desincronización hacía que el "test set" de las figuras del paper no
# fuera el mismo que el usado para entrenar el modelo guardado).
# Si quieres correr sobre CD-HIT dedup, pásalo explícito con --data-file y
# el nombre de salida del modelo lo reflejará (no se pisa el .pkl anterior).
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
N_GROUP_FOLDS       = 5   # folds internos (por organismo) para el objective de Optuna
TEST_ORG_FRACTION   = 0.20

META_COLS  = ['Protein_ID', 'Organism_Source', 'Taxon_ID', 'T_opt_C',
              'Organism_Resolved', 'EC_Class', 'Thermal_Class']
TARGET_COL = 'Thermal_Class'
GROUP_COL  = 'Organism_Source'


def load_and_balance(data_file):
    print("Cargando dataset de features...")
    df = pd.read_csv(data_file)

    if GROUP_COL not in df.columns:
        raise ValueError(
            f"No existe la columna '{GROUP_COL}'. Este script requiere el CSV "
            f"generado por la versión corregida de 03_feature_extraction.py "
            f"(con organismo real, no el bug que guardaba EC_Class)."
        )

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
    """
    Split train/test por ORGANISMO completo, no por proteína.

    Por qué: Thermal_Class es una etiqueta a nivel de organismo (todo el
    proteoma de un psicrófilo se marca Cold). Un split a nivel de proteína
    deja casi seguro al mismo organismo repartido entre train y test, y el
    modelo puede aprender "firma genómica de esta especie" en vez de una
    señal de adaptación al frío que generalice a un genoma nuevo — que es
    exactamente el caso de uso real (09_predict_new_genome.py).

    Este split reserva organismos ENTEROS para test, estratificando por
    Thermal_Class a nivel de organismo (no de proteína) para mantener
    representación de ambas clases en el test set.
    """
    organisms = (df[[GROUP_COL, TARGET_COL]]
                 .drop_duplicates(subset=GROUP_COL)
                 .reset_index(drop=True))
    n_orgs = len(organisms)
    print(f"\n  Organismos únicos en el dataset: {n_orgs}")
    print(f"    Cold : {(organisms[TARGET_COL]==0).sum()}   "
          f"Warm : {(organisms[TARGET_COL]==1).sum()}")

    if n_orgs < 10:
        print(f"  ⚠️  Solo {n_orgs} organismos — el split test tendrá muy pocos "
              f"grupos. Los intervalos de confianza del AUC serán amplios; "
              f"repórtalo así en el paper (N real de replicación biológica = {n_orgs}).")

    train_orgs, test_orgs = train_test_split(
        organisms[GROUP_COL],
        test_size=TEST_ORG_FRACTION,
        stratify=organisms[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    train_orgs, test_orgs = set(train_orgs), set(test_orgs)

    train_df = df[df[GROUP_COL].isin(train_orgs)].reset_index(drop=True)
    test_df  = df[df[GROUP_COL].isin(test_orgs)].reset_index(drop=True)

    print(f"\n  Train: {len(train_df):,} proteínas de {len(train_orgs)} organismos")
    print(f"  Test  : {len(test_df):,} proteínas de {len(test_orgs)} organismos "
          f"(NUNCA vistos en train)")
    print(f"  Organismos en test: {sorted(test_orgs)}")

    assert train_orgs.isdisjoint(test_orgs), "BUG: un organismo quedó en ambos splits"

    feat_cols = [c for c in df.columns if c not in META_COLS]
    X_tr = train_df[feat_cols].astype(np.float32)
    X_te = test_df[feat_cols].astype(np.float32)
    y_tr = train_df[TARGET_COL].values
    y_te = test_df[TARGET_COL].values
    groups_tr = train_df[GROUP_COL].values

    return (X_tr, X_te, y_tr, y_te, groups_tr,
            train_df[META_COLS], test_df[META_COLS], feat_cols)


def legacy_protein_level_split(df, feat_cols):
    """
    Reproduce el split ORIGINAL (aleatorio a nivel de proteína, estratificado
    por EC_Class x Thermal_Class) SOLO para reportarlo como comparación y
    hacer visible cuánto infla el número el leakage por organismo.

    Este número NO debe usarse como métrica principal del paper — se incluye
    para transparencia y para poder mostrar el delta vs. el split correcto.
    """
    d = df.copy()
    d['strat_key'] = d['EC_Class'] + "_" + d[TARGET_COL].astype(str)
    counts = d['strat_key'].value_counts()
    rare = counts[counts < 5].index.tolist()
    if rare:
        d = d[~d['strat_key'].isin(rare)]

    X = d[feat_cols].astype(np.float32)
    y = d[TARGET_COL].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=d['strat_key'], random_state=RANDOM_STATE)
    return X_tr, X_te, y_tr, y_te


def objective(trial, X_tr, y_tr, groups_tr):
    """
    Optuna optimiza el F2 promedio de un GroupKFold POR ORGANISMO calculado
    únicamente sobre datos de train. El test set (organismos held-out) nunca
    entra a esta función — antes, el objective recibía X_te/y_te directamente
    y Optuna elegía los hiperparámetros que mejor rendían en el propio test,
    lo cual invalida esa métrica como estimación de generalización.
    """
    param = {
        'n_estimators':      trial.suggest_int('n_estimators', 150, 500),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth':         trial.suggest_int('max_depth', 4, 10),
        'num_leaves':        trial.suggest_int('num_leaves', 20, 80),
        'scale_pos_weight':  trial.suggest_float('scale_pos_weight', 1.0, 4.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_jobs': 2, 'random_state': RANDOM_STATE, 'verbose': -1,
    }

    n_groups = len(np.unique(groups_tr))
    n_splits = min(N_GROUP_FOLDS, n_groups)
    if n_splits < 2:
        # Muy pocos organismos en train para hacer GroupKFold interno;
        # cae a un único split por organismo como validación interna.
        gkf_splits = [next(GroupKFold(n_splits=2).split(X_tr, y_tr, groups_tr))]
    else:
        gkf_splits = list(GroupKFold(n_splits=n_splits).split(X_tr, y_tr, groups_tr))

    fold_f2s = []
    for tr_idx, val_idx in gkf_splits:
        model = lgb.LGBMClassifier(**param)
        model.fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
        probs_cold = model.predict_proba(X_tr.iloc[val_idx])[:, 0]
        y_val = y_tr[val_idx]
        if len(np.unique(y_val)) < 2:
            continue
        best_f2 = 0.0
        for t in np.linspace(0.1, 0.9, 40):
            f2 = fbeta_score(1 - y_val, (probs_cold >= t).astype(int), beta=2, zero_division=0)
            best_f2 = max(best_f2, f2)
        fold_f2s.append(best_f2)

    return float(np.mean(fold_f2s)) if fold_f2s else 0.0


def find_best_threshold(probs_cold, y_true):
    """Barre umbrales y devuelve el que maximiza F2. OJO: quien llama a esta
    función decide sobre QUÉ probabilidades corre — nunca debe ser el test set
    (ver find_threshold_via_oof_cv, que es la forma correcta de fijar el
    umbral final sin tocar test)."""
    best_t, best_f2 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 100):
        f2 = fbeta_score(1 - y_true, (probs_cold >= t).astype(int), beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    return best_t, best_f2


def find_threshold_via_oof_cv(X_tr, y_tr, groups_tr, params, n_splits=N_GROUP_FOLDS):
    """
    FIX (leakage de umbral): el umbral final de decisión es, en la práctica,
    un hiperparámetro del modelo — igual que n_estimators o max_depth. Antes
    se elegía barriendo probabilidades calculadas sobre el TEST set (ver
    versión anterior de `train()`), lo que sesga optimistamente Precision,
    Recall, F2 y Accuracy reportados como métricas de generalización (el AUC
    no se ve afectado porque es independiente del umbral).

    Aquí el umbral se fija usando SOLO datos de train, vía GroupKFold por
    organismo: se entrena un modelo por fold, se predice sobre el fold de
    validación (out-of-fold), se concatenan esas predicciones OOF de todo
    train, y se busca el umbral F2-óptimo sobre ese conjunto. El test set no
    interviene en ningún momento de esta función.
    """
    n_groups = len(np.unique(groups_tr))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        print("  ⚠️  Muy pocos organismos en train para CV de umbral — "
              "usando 0.5 por defecto.")
        return 0.5

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(y_tr))
    oof_mask  = np.zeros(len(y_tr), dtype=bool)

    for tr_idx, val_idx in gkf.split(X_tr, y_tr, groups_tr):
        if len(np.unique(y_tr[val_idx])) < 2:
            continue  # fold monoclase: no aporta señal útil para el umbral
        fold_model = lgb.LGBMClassifier(**params)
        fold_model.fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X_tr.iloc[val_idx])[:, 0]
        oof_mask[val_idx] = True

    if oof_mask.sum() == 0:
        print("  ⚠️  Sin predicciones OOF válidas — usando 0.5 por defecto.")
        return 0.5

    threshold, f2_oof = find_best_threshold(oof_probs[oof_mask], y_tr[oof_mask])
    print(f"  Umbral fijado vía CV out-of-fold en TRAIN (n={oof_mask.sum():,} "
          f"proteínas OOF): {threshold:.4f}  (F2 OOF = {f2_oof:.4f})")
    return threshold


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
    print("  PsychroScan — Entrenamiento LightGBM + Optuna (v4)")
    print("  Split por ORGANISMO + Optuna sin leakage al test")
    print("=" * 70 + "\n")

    df = load_and_balance(data_file)
    (X_tr, X_te, y_tr, y_te, groups_tr,
     meta_tr, meta_te, feat_cols) = split_by_organism(df)

    # ── Optuna (solo ve train, vía GroupKFold interno) ─────────────────────────
    print(f"\n🔍 Optimizando {OPTUNA_TRIALS} trials (Optuna, GroupKFold interno x{N_GROUP_FOLDS})...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, groups_tr),
        n_trials=OPTUNA_TRIALS, show_progress_bar=True,
    )

    print(f"\n✅ Mejor F2-Score (CV interna, sin tocar test): {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"     {k:<25} = {v}")

    # ── Umbral de decisión: fijado SOLO con train (CV out-of-fold) ────────────
    # FIX: antes se calculaba barriendo las probabilidades del TEST set, lo
    # que sesga Precision/Recall/F2/Accuracy (ver find_threshold_via_oof_cv).
    best_params = {**study.best_params, 'n_jobs': 2,
                   'random_state': RANDOM_STATE, 'verbose': -1}
    print(f"\n🎯 Fijando umbral de decisión (CV out-of-fold en train, sin tocar test)...")
    threshold = find_threshold_via_oof_cv(X_tr, y_tr, groups_tr, best_params)

    # ── Modelo final: se refita en TODO train, se evalúa UNA vez en test ──────
    print("\nEntrenando modelo final sobre todo train...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_tr, y_tr)

    probs_cold = final_model.predict_proba(X_te)[:, 0]
    auc_score  = roc_auc_score(1 - y_te, probs_cold)
    # F2 de test es ahora puramente DIAGNÓSTICO — el umbral ya estaba fijado
    # antes de mirar test, así que este número no está sobreajustado al umbral.
    cold_pred_te = (probs_cold >= threshold).astype(int)          # 1 = predicho Cold
    best_f2 = fbeta_score(1 - y_te, cold_pred_te, beta=2, zero_division=0)
    y_pred  = 1 - cold_pred_te                                    # de vuelta a encoding Thermal_Class (0=cold,1=warm)

    print("\n" + "=" * 70)
    print(f"  RESULTADO PRINCIPAL — Held-out por ORGANISMO (nunca visto en train)")
    print(f"  AUC: {auc_score:.4f}  |  F2-Score Cold (umbral fijado en train): {best_f2:.4f}")
    print(f"  Umbral usado (fijado en train, NO en test): {threshold:.4f}")
    print("=" * 70)
    print(classification_report(y_te, y_pred, target_names=['Cold (0)', 'Warm (1)']))
    cm = confusion_matrix(y_te, y_pred)
    print(f"  Matriz de confusión:")
    print(f"              Pred Cold  Pred Warm")
    print(f"  Real Cold :   {cm[0,0]:>6}     {cm[0,1]:>6}")
    print(f"  Real Warm :   {cm[1,0]:>6}     {cm[1,1]:>6}")

    # ── Comparación con el split legacy (a nivel de proteína) ──────────────────
    legacy_auc = None
    if run_legacy_comparison:
        print("\n" + "-" * 70)
        print("  COMPARACIÓN — split legacy a nivel de proteína (NO usar como métrica")
        print("  principal; se incluye para mostrar cuánto infla el leakage por organismo)")
        print("-" * 70)
        X_tr_l, X_te_l, y_tr_l, y_te_l = legacy_protein_level_split(df, feat_cols)
        legacy_model = lgb.LGBMClassifier(**best_params)
        legacy_model.fit(X_tr_l, y_tr_l)
        legacy_probs = legacy_model.predict_proba(X_te_l)[:, 0]
        legacy_auc = roc_auc_score(1 - y_te_l, legacy_probs)
        delta = legacy_auc - auc_score
        print(f"  AUC (split proteína, organismos mezclados) : {legacy_auc:.4f}")
        print(f"  AUC (split organismo, held-out real)       : {auc_score:.4f}")
        print(f"  Δ atribuible a leakage por organismo       : {delta:+.4f}")

    if auc_score >= 0.85:
        print(f"\n  Nivel publicable por el criterio del proyecto (AUC >= 0.85),")
        print(f"  evaluado correctamente en organismos nunca vistos.")
    else:
        print(f"\n  AUC < 0.85 en held-out por organismo. Esta es la generalización")
        print(f"  real esperable en un genoma nuevo — considera ampliar el dataset")
        print(f"  de organismos (no solo de proteínas) antes de publicar.")

    # ── Guardar modelo + manifiesto de split (fuente única de verdad) ─────────
    joblib.dump(final_model, os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
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
        "threshold_method":     "oof_cv_train_only",  # nunca se tocó el test para elegirlo
    }
    manifest_path = os.path.join(MODELS_DIR, "split_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  📄 Manifiesto de split guardado → {manifest_path}")
    print(f"     TODOS los demás scripts (07,08,09,10,11,blast_benchmark,")
    print(f"     Brenda_validation) deben leer train/test_protein_ids de aquí")
    print(f"     en vez de re-derivar su propio split.")

    # ── Top 15 diversificado por EC (sobre el test held-out por organismo) ────
    top_15 = build_diverse_top15(meta_te, probs_cold, y_te)
    print("\n" + "=" * 70)
    print(f"  TOP 15 DIVERSIFICADO (max {TOP15_MAX_PER_EC} por EC_Class, organismos held-out)")
    print("=" * 70)
    for _, row in top_15.iterrows():
        print(f"  {row['Protein_ID'][:33]:<34} {row['EC_Class']:<22} "
              f"{row['Organism_Source']:<28} {row['Cold_Probability']*100:.2f}%")

    top_15.to_csv(os.path.join(RESULTS_DIR, "top15_candidates_raw.csv"), index=False)
    print(f"\n  Top 15 guardado → results/top15_candidates_raw.csv")
    print("  Siguiente paso  → 07_biological_annotation.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', default=DEFAULT_DATA_FILE,
                        help='CSV de features a usar (default: dataset_features.csv completo)')
    parser.add_argument('--skip-legacy-comparison', action='store_true',
                        help='Omitir el cálculo del split legacy (ahorra tiempo)')
    args = parser.parse_args()
    train(args.data_file, run_legacy_comparison=not args.skip_legacy_comparison)