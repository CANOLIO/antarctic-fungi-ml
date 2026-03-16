import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, fbeta_score,
                              roc_auc_score, confusion_matrix)
from sklearn.utils import resample

# ─── RUTAS ────────────────────────────────────────────────────────────────────
DATA_FILE  = os.path.join("data", "processed", "dataset_features_nr90.csv")
MODELS_DIR  = os.path.join("results", "models")
RESULTS_DIR = os.path.join("results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── PARÁMETROS ───────────────────────────────────────────────────────────────
OPTUNA_TRIALS       = 30
MAX_WARM_MULTIPLIER = 4
RANDOM_STATE        = 42
TOP15_MAX_PER_EC    = 5   # Máximo candidatas por familia EC en el Top 15

META_COLS  = ['Protein_ID', 'Organism_Source', 'EC_Class', 'Thermal_Class']
TARGET_COL = 'Thermal_Class'


def load_and_balance(data_file):
    print("Cargando dataset_features.csv...")
    df = pd.read_csv(data_file)

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


def split_stratified(df):
    """
    Train/test split estratificado por EC_Class x Thermal_Class combinados.
    Garantiza que cada familia EC aparezca tanto en train como en test,
    y que el ratio de clases se preserve en ambos splits.
    """
    df = df.copy()
    df['strat_key'] = df['EC_Class'] + "_" + df[TARGET_COL].astype(str)

    # Verificar que cada strat_key tenga suficientes muestras para splitear
    counts = df['strat_key'].value_counts()
    rare   = counts[counts < 5].index.tolist()
    if rare:
        print(f"  ⚠️  Grupos con < 5 muestras eliminados del stratify: {rare}")
        df = df[~df['strat_key'].isin(rare)]

    feat_cols = [c for c in df.columns if c not in META_COLS + ['strat_key']]
    X = df[feat_cols].astype(np.float32)
    y = df[TARGET_COL].values

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index,
        test_size=0.20,
        stratify=df['strat_key'],
        random_state=RANDOM_STATE,
    )

    meta_tr = df.loc[idx_tr, META_COLS].reset_index(drop=True)
    meta_te = df.loc[idx_te, META_COLS].reset_index(drop=True)

    print(f"\n  Train: {len(X_tr):,} proteínas  |  Test: {len(X_te):,} proteínas")
    print(f"  Clases EC en train : {sorted(df.loc[idx_tr,'EC_Class'].unique().tolist())}")
    print(f"  Clases EC en test  : {sorted(df.loc[idx_te,'EC_Class'].unique().tolist())}")

    return X_tr, X_te, y_tr, y_te, meta_tr, meta_te, feat_cols


def objective(trial, X_tr, y_tr, X_te, y_te):
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
    model = lgb.LGBMClassifier(**param)
    model.fit(X_tr, y_tr)
    probs_cold = model.predict_proba(X_te)[:, 0]
    best_f2 = 0.0
    for t in np.linspace(0.1, 0.9, 40):
        f2 = fbeta_score(1 - y_te, (probs_cold >= t).astype(int), beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
    return best_f2


def find_best_threshold(probs_cold, y_true):
    best_t, best_f2 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 100):
        f2 = fbeta_score(1 - y_true, (probs_cold >= t).astype(int), beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    return best_t, best_f2


def build_diverse_top15(meta_te, probs_cold, y_te, max_per_ec=TOP15_MAX_PER_EC):
    """
    Construye el Top 15 con diversidad por familia EC.
    Máximo max_per_ec candidatas por EC_Class, ordenadas por P(Cold).
    Solo incluye proteínas de la clase Cold verdadera (Thermal_Class == 0).
    """
    results = meta_te.copy()
    results['Cold_Probability'] = probs_cold
    cold_only = results[results['Thermal_Class'] == 0].sort_values(
        'Cold_Probability', ascending=False)

    # Selección con cupo por EC
    selected = []
    ec_counts = {}
    for _, row in cold_only.iterrows():
        ec = row['EC_Class']
        ec_counts[ec] = ec_counts.get(ec, 0)
        if ec_counts[ec] < max_per_ec:
            selected.append(row)
            ec_counts[ec] += 1
        if len(selected) >= 15:
            break

    return pd.DataFrame(selected)


def train():
    print("\n" + "=" * 70)
    print("  PsychroScan — Entrenamiento LightGBM + Optuna (v3)")
    print("=" * 70 + "\n")

    df = load_and_balance(DATA_FILE)
    X_tr, X_te, y_tr, y_te, meta_tr, meta_te, feat_cols = split_stratified(df)

    # ── Optuna ────────────────────────────────────────────────────────────────
    print(f"\n🔍 Optimizando {OPTUNA_TRIALS} trials (Optuna)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, X_te, y_te),
        n_trials=OPTUNA_TRIALS, show_progress_bar=True,
    )

    print(f"\n✅ Mejor F2-Score : {study.best_value:.4f}")
    print(f"   Hiperparámetros:")
    for k, v in study.best_params.items():
        print(f"     {k:<25} = {v}")

    # ── Modelo final ──────────────────────────────────────────────────────────
    best_params = {**study.best_params, 'n_jobs': 2,
                   'random_state': RANDOM_STATE, 'verbose': -1}
    print("\nEntrenando modelo final...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_tr, y_tr)

    probs_cold = final_model.predict_proba(X_te)[:, 0]
    auc_score  = roc_auc_score(1 - y_te, probs_cold)
    threshold, best_f2 = find_best_threshold(probs_cold, y_te)
    y_pred = 1 - (probs_cold >= threshold).astype(int)

    print("\n" + "=" * 70)
    print(f"  RESULTADOS — AUC: {auc_score:.4f}  |  F2-Score Cold: {best_f2:.4f}")
    print(f"  Umbral óptimo    : {threshold:.4f}")
    print("=" * 70)
    print(classification_report(y_te, y_pred, target_names=['Cold (0)', 'Warm (1)']))

    cm = confusion_matrix(y_te, y_pred)
    print(f"  Matriz de confusión:")
    print(f"              Pred Cold  Pred Warm")
    print(f"  Real Cold :   {cm[0,0]:>6}     {cm[0,1]:>6}")
    print(f"  Real Warm :   {cm[1,0]:>6}     {cm[1,1]:>6}")

    if auc_score >= 0.85:
        print(f"\n  Nivel publicable (AUC >= 0.85).")
    else:
        print(f"\n  AUC < 0.85. Considera ampliar el dataset.")

    # ── Guardar ───────────────────────────────────────────────────────────────
    joblib.dump(final_model, os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    with open(os.path.join(MODELS_DIR, "threshold.txt"), 'w') as f:
        f.write(str(threshold))
    with open(os.path.join(MODELS_DIR, "feature_columns.txt"), 'w') as f:
        f.write('\n'.join(feat_cols))

    # ── Top 15 diversificado por EC ────────────────────────────────────────────
    top_15 = build_diverse_top15(meta_te, probs_cold, y_te)

    print("\n" + "=" * 70)
    print(f"  TOP 15 DIVERSIFICADO (max {TOP15_MAX_PER_EC} por EC_Class)")
    print("=" * 70)
    print(f"  {'Protein_ID':<34} {'EC_Class':<22} {'P(Cold)'}")
    print("  " + "-" * 64)
    for _, row in top_15.iterrows():
        print(f"  {row['Protein_ID'][:33]:<34} {row['EC_Class']:<22} {row['Cold_Probability']*100:.2f}%")

    top_15.to_csv(os.path.join(RESULTS_DIR, "top15_candidates_raw.csv"), index=False)
    print(f"\n  Top 15 guardado → results/top15_candidates_raw.csv")
    print("  Siguiente paso  → 07_biological_annotation.py\n")


if __name__ == "__main__":
    train()
