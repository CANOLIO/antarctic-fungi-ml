import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import hypergeom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
MODELS_DIR    = os.path.join("results", "models")
DATA_FILE     = os.path.join("data", "processed", "dataset_features.csv")
MANIFEST_FILE = os.path.join(MODELS_DIR, "split_manifest.json")
FIGURES_DIR   = os.path.join("results", "figures")
BENCHMARK_DIR = os.path.join("results", "benchmark")
os.makedirs(FIGURES_DIR, exist_ok=True)

META_COLS = ['Protein_ID', 'Organism_Source', 'Taxon_ID', 'T_opt_C',
             'Organism_Resolved', 'EC_Class', 'Thermal_Class']

C_COLD   = "#4a90d9"
C_WARM   = "#e07b54"
C_MODEL  = "#2ecc71"
C_GRAVY  = "#95a5a6"
C_LOGREG = "#f39c12"
GREY     = "#ecf0f1"


def load_model_and_data():
    """
    FIX: antes esta función devolvía X, y del dataset COMPLETO, y cada figura
    (fig_baseline_comparison, ablation_thermoadaptive) hacía su PROPIO
    train_test_split aleatorio a nivel de proteína e independiente entre sí
    y de 05_train_model.py (comentario original: "Split reproducible (mismo
    seed que 05)" — pero ni el archivo de datos ni la clave de estratificación
    coincidían con 05). Con alta probabilidad, parte de ese "test" ya había
    sido visto por el modelo durante el entrenamiento.

    Ahora se carga el manifiesto de split por organismo generado por
    05_train_model.py y se devuelve el split EXACTO (mismos Protein_ID) que
    de verdad separó train/test al entrenar el modelo guardado.
    """
    model = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    fcols = open(os.path.join(MODELS_DIR, "feature_columns.txt")).read().strip().split('\n')

    if not os.path.exists(MANIFEST_FILE):
        raise FileNotFoundError(
            f"No existe {MANIFEST_FILE}. Corre primero 05_train_model.py (v4) "
            f"— sin el manifiesto, cualquier split que se re-derive aquí puede "
            f"no coincidir con el que realmente entrenó el modelo guardado."
        )
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    df = pd.read_csv(DATA_FILE)
    train_ids = set(manifest['train_protein_ids'])
    test_ids  = set(manifest['test_protein_ids'])

    train_df = df[df['Protein_ID'].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df['Protein_ID'].isin(test_ids)].reset_index(drop=True)

    X_tr = train_df[fcols].astype(np.float32)
    X_te = test_df[fcols].astype(np.float32)
    y_tr = train_df['Thermal_Class'].values
    y_te = test_df['Thermal_Class'].values

    # X, y "completos" se mantienen solo para fig_feature_importance (que no
    # evalúa el modelo, solo lee model.feature_importances_ ya calculado).
    X_full = df[fcols].astype(np.float32)
    y_full = df['Thermal_Class'].values

    return model, fcols, X_full, y_full, X_tr, X_te, y_tr, y_te, manifest


# ══════════════════════════════════════════════════════════════════════════════
# FIG E — Feature Importance Top 30 (no requiere split; lee el modelo ya entrenado)
# ══════════════════════════════════════════════════════════════════════════════
def fig_feature_importance(model, feat_cols):
    print("  Generando Fig E — Feature Importance...")

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature':    feat_cols,
        'Importance': importances,
    }).sort_values('Importance', ascending=False).head(30)

    def categorize(f):
        if f in ('IVYWREL_Index', 'CvP_Bias', 'Flexibility_Ratio'):
            return 'Thermoadaptive'
        if f.startswith('DPC_'):
            return 'Dipeptide'
        if f.startswith('AAC_'):
            return 'Amino Acid'
        return 'Physicochemical'

    feat_df['Category'] = feat_df['Feature'].apply(categorize)

    color_map = {
        'Thermoadaptive': '#e74c3c',
        'Dipeptide':      '#3498db',
        'Amino Acid':     '#2ecc71',
        'Physicochemical':'#95a5a6',
    }
    colors = feat_df['Category'].map(color_map)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(range(len(feat_df)), feat_df['Importance'].values,
            color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(feat_df)))
    ax.set_yticklabels(feat_df['Feature'].values, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (LightGBM gain)', fontsize=10)
    ax.set_title('Top 30 Features — PsychroScan Model\n'
                 'Top features are mathematical components of thermoadaptive indices',
                 fontsize=11, fontweight='bold')
    ax.set_facecolor(GREY)
    ax.grid(axis='x', alpha=0.4, color='white')

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

    for i, (_, row) in enumerate(feat_df.iterrows()):
        if row['Category'] == 'Thermoadaptive':
            ax.text(row['Importance'] + max(feat_df['Importance']) * 0.01,
                    i, row['Feature'], va='center', fontsize=7.5,
                    color='#c0392b', fontweight='bold')

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, '11E_Feature_Importance.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"     → {out}")
    return feat_df


# ══════════════════════════════════════════════════════════════════════════════
# FIG F — Comparación AUC: LightGBM vs GRAVY baseline vs Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════
def fig_baseline_comparison(model, feat_cols, X_tr, X_te, y_tr, y_te):
    """
    FIX: ahora recibe el split EXACTO del manifiesto (organismos held-out
    nunca vistos por el modelo entrenado) en vez de re-derivar un split
    propio. Esta es la figura que produce el AUC citado como headline —
    con el fix, el número refleja generalización real a organismos nuevos.
    """
    print("  Generando Fig F — Baseline Comparison (organismos held-out)...")

    probs_lgbm      = model.predict_proba(X_te.astype(np.float32))[:, 0]
    fpr_l, tpr_l, _ = roc_curve(1 - y_te, probs_lgbm)
    auc_lgbm        = roc_auc_score(1 - y_te, probs_lgbm)

    gravy_idx = feat_cols.index('GRAVY') if 'GRAVY' in feat_cols else None
    if gravy_idx is not None:
        gravy_scores    = -X_te.iloc[:, gravy_idx].values
        fpr_g, tpr_g, _ = roc_curve(1 - y_te, gravy_scores)
        auc_gravy       = roc_auc_score(1 - y_te, gravy_scores)
    else:
        fpr_g, tpr_g, auc_gravy = None, None, None

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)
    lr       = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_tr_sc, y_tr)
    probs_lr        = lr.predict_proba(X_te_sc)[:, 0]
    fpr_lr, tpr_lr, _ = roc_curve(1 - y_te, probs_lr)
    auc_lr          = roc_auc_score(1 - y_te, probs_lr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_l,  tpr_l,  color=C_MODEL,  lw=2.5,
            label=f'LightGBM + Optuna  (AUC = {auc_lgbm:.4f})')
    ax.plot(fpr_lr, tpr_lr, color=C_LOGREG, lw=1.8, linestyle='--',
            label=f'Logistic Regression (AUC = {auc_lr:.4f})')
    if fpr_g is not None:
        ax.plot(fpr_g, tpr_g, color=C_GRAVY, lw=1.8, linestyle=':',
                label=f'GRAVY Threshold     (AUC = {auc_gravy:.4f})')
    ax.plot([0,1],[0,1], color='#bdc3c7', lw=1, linestyle='--', label='Random (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves — Model vs Baselines\n'
                 'Cold-active enzyme classification (held-out organisms)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9.5)
    ax.set_facecolor(GREY)
    ax.grid(alpha=0.4, color='white')

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, '11F_Baseline_Comparison.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    gravy_str = f"{auc_gravy:.4f}" if auc_gravy is not None else "N/A"
    print(f"     → {out}")
    print(f"     [LightGBM={auc_lgbm:.4f} | LogReg={auc_lr:.4f} | GRAVY={gravy_str}]  (held-out por organismo)")

    return {'LightGBM': auc_lgbm, 'LogisticRegression': auc_lr, 'GRAVY': auc_gravy}


# ══════════════════════════════════════════════════════════════════════════════
# TABLA — Test hipergeométrico por organismo (no afectado por el split; opera
# sobre benchmarks externos generados por benchmark_known_enzymes.py)
# ══════════════════════════════════════════════════════════════════════════════
def table_hypergeometric():
    from scipy.stats import fisher_exact
    print("  Calculando tests hipergeométricos...")

    benchmark_files = [
        f for f in os.listdir(BENCHMARK_DIR)
        if f.endswith('_benchmark.csv') and f != 'blast_benchmark.csv'
    ]
    if not benchmark_files:
        print("     ⚠️  Sin archivos de benchmark. Corre benchmark_known_enzymes.py primero.")
        return None

    rows = []
    for bf in sorted(benchmark_files):
        organism = bf.replace('_benchmark.csv', '').replace('_', ' ')
        df = pd.read_csv(os.path.join(BENCHMARK_DIR, bf))

        N = len(df)
        K = df['Enzyme_Category'].notna().sum()

        k15  = int((df['Enzyme_Category'].notna() & (df['Rank'] <= 15)).sum())
        ct15 = np.array([[k15, K - k15], [15 - k15, N - K - (15 - k15)]])
        _, p15 = fisher_exact(ct15, alternative='greater')
        exp15  = round(K * 15 / N, 2)
        fold15 = round(k15 / max(exp15, 0.01), 2)

        n1   = max(int(np.ceil(N * 0.01)), 1)
        k1   = int((df['Enzyme_Category'].notna() & (df['Rank'] <= n1)).sum())
        p1   = hypergeom.sf(k1 - 1, N, K, n1)
        exp1 = round(K * 0.01, 2)

        n5   = int(np.ceil(N * 0.05))
        k5   = int((df['Enzyme_Category'].notna() & (df['Rank'] <= n5)).sum())
        p5   = hypergeom.sf(k5 - 1, N, K, n5)
        exp5 = round(K * 0.05, 2)

        def sig(p):
            return "✅" if p < 0.05 else ("~" if p < 0.10 else "❌")

        rows.append({
            'Organism': organism, 'N': N, 'K_hydrolytic': K,
            'k_top15': k15, 'expected_top15': exp15, 'fold_top15': fold15,
            'p_top15_Fisher': f"{p15:.2e}", 'sig_top15': sig(p15),
            'k_top1pct': k1, 'expected_top1pct': exp1, 'p_top1pct': f"{p1:.2e}", 'sig_top1pct': sig(p1),
            'k_top5pct': k5, 'expected_top5pct': exp5, 'p_top5pct': f"{p5:.2e}", 'sig_top5pct': sig(p5),
        })

    table_df = pd.DataFrame(rows)
    out = os.path.join(BENCHMARK_DIR, 'hypergeometric_table.csv')
    table_df.to_csv(out, index=False)

    print(f"\n  ┌─ ENRIQUECIMIENTO DE ENZIMAS HIDROLÍTICAS")
    for _, r in table_df.iterrows():
        print(f"  │  {r['Organism']:<40} "
              f"k={r['k_top15']} exp={r['expected_top15']} ({r['fold_top15']}x) "
              f"{r['p_top15_Fisher']:>9} {r['sig_top15']}  "
              f"k={r['k_top1pct']} {r['p_top1pct']:>9} {r['sig_top1pct']}  "
              f"k={r['k_top5pct']} {r['p_top5pct']:>9} {r['sig_top5pct']}")
    print(f"  │  Sig: ✅ p<0.05  ~ p<0.10  ❌ p≥0.10")
    print(f"  Tabla → {out}\n")
    return table_df


# ══════════════════════════════════════════════════════════════════════════════
# FIG G — Leave-One-Organism-Out Cross Validation
# ══════════════════════════════════════════════════════════════════════════════
def fig_looo_cv(feat_cols, n_folds: int = 5):
    """
    Cross-validation agrupada por organismo (StratifiedGroupKFold).

    IMPORTANTE — por qué esto NO es un literal "leave-one-organism-out":
    Thermal_Class es una etiqueta a nivel de ORGANISMO completo (todo el
    proteoma de un psicrófilo es Cold). Eso significa que un fold que deja
    fuera UN SOLO organismo tiene, por construcción, una sola clase en test
    — el AUC-ROC no se puede calcular (se necesitan ambas clases). Lo probé
    con LeaveOneGroupOut real: los ~20-27 folds se saltan TODOS por este
    motivo. Esto no es un bug de código, es una propiedad estructural del
    dataset que solo se revela al corregir Organism_Source.

    (Nota histórica: la versión anterior de este archivo intentaba LOOO
    literal, pero como Organism_Source contenía el nombre de la clase EC
    en vez del organismo real — el bug de 03_feature_extraction.py — nunca
    llegó a manifestarse este problema: con solo 5 grupos "EC" cada uno
    mezclaba ambas clases y el AUC sí se podía calcular. El título del
    gráfico incluso tenía dos líneas contradictorias: 'Leave-One-Organism-Out'
    y, justo debajo, 'Leave-One-EC-Out' — sobreescrita a último momento
    cuando se notó la discrepancia, sin corregir la causa raíz.)

    La alternativa correcta: StratifiedGroupKFold — varios organismos por
    fold (nunca repartidos entre train/test del mismo fold), balanceando
    Cold/Warm por fold. Cada fold sigue siendo un test genuino sobre
    organismos nunca vistos en ese fold de entrenamiento, pero con
    suficiente diversidad de clase para poder calcular AUC.
    """
    print(f"  Generando Fig G — CV agrupada por organismo (StratifiedGroupKFold, {n_folds} folds)...")

    df = pd.read_csv(DATA_FILE)
    if 'Organism_Source' not in df.columns:
        print("     ⚠️  Columna Organism_Source no encontrada.")
        return None

    X   = df[feat_cols].astype(np.float32)
    y   = df['Thermal_Class'].values
    grp = df['Organism_Source'].values

    n_orgs = len(np.unique(grp))
    print(f"     Organismos únicos: {n_orgs} | Proteínas: {len(df):,}")
    if n_orgs < n_folds * 2:
        print(f"     ⚠️  Pocos organismos ({n_orgs}) para {n_folds} folds — "
              f"considera reducir n_folds.")

    sgkf    = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    best_params = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl")).get_params()

    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, grp)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        test_orgs  = sorted(set(grp[test_idx]))

        if len(np.unique(y_te)) < 2:
            print(f"     [Fold {fold_i+1}/{n_folds}] {len(test_orgs)} organismos — "
                  f"omitido (una sola clase en el fold)")
            continue

        model_fold = lgb.LGBMClassifier(**best_params)
        model_fold.fit(X_tr, y_tr)
        probs = model_fold.predict_proba(X_te.astype(np.float32))[:, 0]

        try:
            auc = roc_auc_score(1 - y_te, probs)
        except Exception:
            auc = float('nan')

        preview = ', '.join(test_orgs[:3]) + ('...' if len(test_orgs) > 3 else '')
        print(f"     [Fold {fold_i+1}/{n_folds}] {len(test_orgs)} organismos ({preview}) AUC={auc:.4f}")
        results.append({'Fold': fold_i + 1, 'Test_Organisms': '; '.join(test_orgs),
                        'N_organisms': len(test_orgs), 'N_proteins': len(test_idx),
                        'AUC': auc})

    if not results:
        print("     ❌ Sin resultados válidos (todos los folds monoclase — "
              "prueba con menos folds o revisa el balance de organismos).")
        return None

    res_df   = pd.DataFrame(results).dropna(subset=['AUC'])
    mean_auc = res_df['AUC'].mean()
    std_auc  = res_df['AUC'].std()
    min_auc  = res_df['AUC'].min()

    fig, ax = plt.subplots(figsize=(10, max(6, len(res_df) * 0.9)))
    ax.barh(range(len(res_df)), res_df['AUC'].values,
            color=C_COLD, edgecolor='white', linewidth=0.5, alpha=0.85)
    for i, (_, row) in enumerate(res_df.iterrows()):
        ax.text(row['AUC'] + 0.005, i, f"{row['AUC']:.4f}", va='center', fontsize=9, color='#2c3e50')
    ax.axvline(mean_auc, color='#2c3e50', lw=2, linestyle='--',
               label=f'Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}')
    ax.axvline(0.85, color='#e74c3c', lw=1.5, linestyle=':', label='Publication threshold (0.85)')
    ax.axvline(0.5,  color='#bdc3c7', lw=1,   linestyle='-', label='Random (0.50)')

    ax.set_yticks(range(len(res_df)))
    ax.set_yticklabels([f"Fold {r['Fold']} (n={r['N_organisms']} organismos)"
                        for _, r in res_df.iterrows()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('AUC-ROC (grupo de organismos held-out por fold)', fontsize=10)
    ax.set_title('Cross-Validation agrupada por organismo (StratifiedGroupKFold)\n'
                 'Cada fold: organismos nunca vistos en el train de ese fold',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_facecolor(GREY)
    ax.grid(axis='x', alpha=0.4, color='white')
    ax.legend(loc='lower right', fontsize=8.5)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, '11G_LOOO_CV.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    csv_out = os.path.join(BENCHMARK_DIR, 'looo_cv_results.csv')
    res_df.to_csv(csv_out, index=False)

    print(f"\n     ✅ CV agrupada por organismo completada:")
    print(f"        AUC medio : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"        AUC mínimo: {min_auc:.4f}")
    print(f"        Folds evaluados: {len(res_df)}/{n_folds}")
    print(f"        → {out}\n        → {csv_out}")

    return res_df, mean_auc, std_auc


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY — Contribución incremental de los índices termoadaptativos
# ══════════════════════════════════════════════════════════════════════════════
def ablation_thermoadaptive(X_tr, X_te, y_tr, y_te, fcols):
    """
    FIX: ahora recibe el split del manifiesto (organismos held-out) en vez
    de re-derivar su propio train_test_split aleatorio a nivel de proteína.
    """
    print("  Ablation study — índices termoadaptativos (organismos held-out)...")

    model  = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    params = {k: v for k, v in model.get_params().items()
              if k not in ('n_jobs', 'random_state', 'verbose')}
    params.update({'n_jobs': 2, 'random_state': 42, 'verbose': -1})

    THERMO = ['IVYWREL_Index', 'CvP_Bias', 'Flexibility_Ratio']
    AAC    = [c for c in fcols if c.startswith('AAC_')]
    PHYSIO = ['Length', 'Molecular_Weight', 'GRAVY', 'Instability_Index',
              'Aromaticity', 'Helix_Fraction', 'Turn_Fraction', 'Sheet_Fraction']

    feature_sets = {
        'Full (431)':              fcols,
        'Without thermo (428)':    [c for c in fcols if c not in THERMO],
        'AAC + physico only (27)': [c for c in fcols if c in AAC + PHYSIO],
    }

    results = {}
    for name, cols in feature_sets.items():
        cols_present = [c for c in cols if c in fcols]
        X_tr_sub = X_tr[cols_present]
        X_te_sub = X_te[cols_present]

        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr_sub, y_tr)
        probs = m.predict_proba(X_te_sub.astype(np.float32))[:, 0]
        auc   = roc_auc_score(1 - y_te, probs)
        results[name] = auc
        print(f"     {name:<30} AUC = {auc:.6f}")

    delta_thermo = results['Full (431)'] - results['Without thermo (428)']
    delta_dpc    = results['Without thermo (428)'] - results['AAC + physico only (27)']
    print(f"\n     Δ AUC por índices termoadaptativos : {delta_thermo:+.6f}")
    print(f"     Δ AUC por dipéptidos               : {delta_dpc:+.6f}")

    if abs(delta_thermo) < 0.0001:
        print(f"\n     → Índices termoadaptativos no aportan AUC incremental detectable.")
        print(f"       Argumento para el paper: valor interpretativo, no predictivo incremental.")
    else:
        print(f"\n     → Índices termoadaptativos aportan Δ AUC = {delta_thermo:+.6f}.")

    abl_df = pd.DataFrame([{'Feature set': k, 'N features': len(feature_sets[k]), 'AUC': v}
                          for k, v in results.items()])
    out = os.path.join(BENCHMARK_DIR, 'ablation_study.csv')
    abl_df.to_csv(out, index=False)
    print(f"     → {out}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--looo', action='store_true', help='Ejecutar LOOO-CV genuino (lento, ~30-60 min)')
    parser.add_argument('--skip-baseline', action='store_true', help='Omitir comparación de baselines (rápido)')
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  PsychroScan — Figuras para Paper (11) v3")
    print("  Split por organismo (manifiesto) + LOOO genuino")
    print("=" * 65 + "\n")

    model, feat_cols, X_full, y_full, X_tr, X_te, y_tr, y_te, manifest = load_model_and_data()
    print(f"  Test held-out: {len(manifest['test_organisms'])} organismos, {len(X_te)} proteínas\n")

    feat_df = fig_feature_importance(model, feat_cols)

    if not args.skip_baseline:
        auc_dict = fig_baseline_comparison(model, feat_cols, X_tr, X_te, y_tr, y_te)
        ablation_thermoadaptive(X_tr, X_te, y_tr, y_te, feat_cols)
    else:
        print("  Fig F omitida (--skip-baseline)")
        auc_dict = None

    hyper_df = table_hypergeometric()

    looo_result = None
    if args.looo:
        looo_result = fig_looo_cv(feat_cols)
    else:
        print("  Fig G (LOOO-CV genuino) omitida — usa --looo para ejecutarla")
        print("  Tiempo estimado: 30-60 min (ahora ~20-27 folds, antes ~5)\n")

    print("\n" + "=" * 65)
    print("  RESUMEN PARA EL PAPER")
    print("=" * 65)

    if auc_dict:
        print(f"  LightGBM AUC (held-out organismo) : {auc_dict['LightGBM']:.4f}")
        print(f"  LogReg AUC   (held-out organismo) : {auc_dict['LogisticRegression']:.4f}")
        if auc_dict['GRAVY']:
            print(f"  GRAVY AUC    (held-out organismo) : {auc_dict['GRAVY']:.4f}")
            print(f"  Δ vs LogReg  : +{auc_dict['LightGBM'] - auc_dict['LogisticRegression']:.4f}")
            print(f"  Δ vs GRAVY   : +{auc_dict['LightGBM'] - auc_dict['GRAVY']:.4f}")

    top3_feats = feat_df.head(3)['Feature'].tolist()
    thermo_in_top30 = feat_df[feat_df['Category'] == 'Thermoadaptive']
    print(f"\n  Top 3 features    : {', '.join(top3_feats)}")
    if len(thermo_in_top30) > 0:
        names = thermo_in_top30['Feature'].tolist()
        ranks = [feat_df.index.get_loc(i) + 1 for i in thermo_in_top30.index]
        print(f"  Thermoadaptive en top 30: {list(zip(names, ranks))}")

    if looo_result is not None:
        _, mean_auc, std_auc = looo_result
        print(f"\n  LOOO-CV (genuino) AUC : {mean_auc:.4f} ± {std_auc:.4f}")
        if mean_auc >= 0.85:
            print(f"  ✅ Generalización validada (≥ 0.85)")
        else:
            print(f"  ⚠️  AUC < 0.85 — este es el número honesto de generalización a "
                  f"organismos nuevos; revisa el dataset antes de publicar.")

    print("\n  Figuras generadas en results/figures/")
    print("  Datos en results/benchmark/\n")


if __name__ == "__main__":
    main()
