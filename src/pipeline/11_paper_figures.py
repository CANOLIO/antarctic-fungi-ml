import os
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
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
MODELS_DIR    = os.path.join("results", "models")
DATA_FILE     = os.path.join("data", "processed", "dataset_features.csv")
FIGURES_DIR   = os.path.join("results", "figures")
BENCHMARK_DIR = os.path.join("results", "benchmark")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── COLORES CONSISTENTES CON EL RESTO DEL PIPELINE ──────────────────────────
C_COLD   = "#4a90d9"
C_WARM   = "#e07b54"
C_MODEL  = "#2ecc71"
C_GRAVY  = "#95a5a6"
C_LOGREG = "#f39c12"
GREY     = "#ecf0f1"


def load_model_and_data():
    model  = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    fcols  = open(os.path.join(MODELS_DIR, "feature_columns.txt")).read().strip().split('\n')
    df     = pd.read_csv(DATA_FILE)
    X      = df[fcols].astype(np.float32)
    y      = df['Thermal_Class'].values
    return model, fcols, X, y


# ══════════════════════════════════════════════════════════════════════════════
# FIG E — Feature Importance Top 30
# ══════════════════════════════════════════════════════════════════════════════
def fig_feature_importance(model, feat_cols):
    print("  Generando Fig E — Feature Importance...")

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature':    feat_cols,
        'Importance': importances,
    }).sort_values('Importance', ascending=False).head(30)

    # Categorizar features para colorear barras
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
    bars = ax.barh(range(len(feat_df)), feat_df['Importance'].values,
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

    legend_patches = [mpatches.Patch(color=v, label=k)
                      for k, v in color_map.items()]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

    # Anotar los tres termoadaptativos si aparecen en top 30
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
def fig_baseline_comparison(model, feat_cols, X, y):
    print("  Generando Fig F — Baseline Comparison...")

    # Split reproducible (mismo seed que 05)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # ── LightGBM (modelo entrenado) ───────────────────────────────────────────
    probs_lgbm   = model.predict_proba(X_te.astype(np.float32))[:, 0]
    fpr_l, tpr_l, _ = roc_curve(1 - y_te, probs_lgbm)
    auc_lgbm     = roc_auc_score(1 - y_te, probs_lgbm)

    # ── Baseline 1: GRAVY score (threshold único) ─────────────────────────────
    # GRAVY negativo → más hidrofílico → correlaciona con frío
    gravy_idx    = feat_cols.index('GRAVY') if 'GRAVY' in feat_cols else None
    if gravy_idx is not None:
        gravy_scores = -X_te.iloc[:, gravy_idx].values  # negado: más neg = más frío
        fpr_g, tpr_g, _ = roc_curve(1 - y_te, gravy_scores)
        auc_gravy    = roc_auc_score(1 - y_te, gravy_scores)
    else:
        fpr_g, tpr_g, auc_gravy = None, None, None

    # ── Baseline 2: Logistic Regression (mismos features) ────────────────────
    scaler       = StandardScaler()
    X_tr_sc      = scaler.fit_transform(X_tr)
    X_te_sc      = scaler.transform(X_te)
    lr           = LogisticRegression(max_iter=500, random_state=42, n_jobs=2)
    lr.fit(X_tr_sc, y_tr)
    probs_lr     = lr.predict_proba(X_te_sc)[:, 0]
    fpr_lr, tpr_lr, _ = roc_curve(1 - y_te, probs_lr)
    auc_lr       = roc_auc_score(1 - y_te, probs_lr)

    # ── Plot ──────────────────────────────────────────────────────────────────
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
                 'Cold-active enzyme classification',
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
    print(f"     [LightGBM={auc_lgbm:.4f} | LogReg={auc_lr:.4f} | GRAVY={gravy_str}]")

    return {'LightGBM': auc_lgbm, 'LogisticRegression': auc_lr,
            'GRAVY': auc_gravy}


# ══════════════════════════════════════════════════════════════════════════════
# TABLA — Test hipergeométrico por organismo
# ══════════════════════════════════════════════════════════════════════════════
def table_hypergeometric():
    """
    Calcula enriquecimiento de enzimas hidrolíticas en tres umbrales:
      - Top 15 (Fisher exact test, 2x2) — el más relevante para el paper
      - Top 1% del proteoma (hipergeométrico)
      - Top 5% del proteoma (hipergeométrico, referencia)

    El Top 15 es el umbral operacional del pipeline — es lo que el usuario
    recibe como output. Fisher exact es el test correcto para conteos pequeños.
    """
    from scipy.stats import fisher_exact
    print("  Calculando tests hipergeométricos...")

    benchmark_files = [f for f in os.listdir(BENCHMARK_DIR)
                       if f.endswith('_benchmark.csv')]
    if not benchmark_files:
        print("     ⚠️  Sin archivos de benchmark. Corre benchmark_known_enzymes.py primero.")
        return None

    rows = []
    for bf in sorted(benchmark_files):
        organism = bf.replace('_benchmark.csv', '').replace('_', ' ')
        df = pd.read_csv(os.path.join(BENCHMARK_DIR, bf))

        N = len(df)
        K = df['Enzyme_Category'].notna().sum()   # hidrolíticas totales

        # ── Top 15: Fisher exact (2x2) ────────────────────────────────────────
        # Tabla:  [hidrolítica ∩ top15]  [hidrolítica ∩ fuera_top15]
        #         [otra ∩ top15]         [otra ∩ fuera_top15]
        k15  = int((df['Enzyme_Category'].notna() & (df['Rank'] <= 15)).sum())
        ct15 = np.array([[k15,        K - k15],
                         [15 - k15,   N - K - (15 - k15)]])
        _, p15 = fisher_exact(ct15, alternative='greater')
        exp15  = round(K * 15 / N, 2)
        fold15 = round(k15 / max(exp15, 0.01), 2)

        # ── Top 1%: hipergeométrico ───────────────────────────────────────────
        n1   = max(int(np.ceil(N * 0.01)), 1)
        k1   = int((df['Enzyme_Category'].notna() & (df['Rank'] <= n1)).sum())
        p1   = hypergeom.sf(k1 - 1, N, K, n1)
        exp1 = round(K * 0.01, 2)

        # ── Top 5%: hipergeométrico ───────────────────────────────────────────
        n5   = int(np.ceil(N * 0.05))
        k5   = int((df['Enzyme_Category'].notna() & (df['Rank'] <= n5)).sum())
        p5   = hypergeom.sf(k5 - 1, N, K, n5)
        exp5 = round(K * 0.05, 2)

        def sig(p):
            return "✅" if p < 0.05 else ("~" if p < 0.10 else "❌")

        rows.append({
            'Organism':         organism,
            'N':                N,
            'K_hydrolytic':     K,
            # Top 15
            'k_top15':          k15,
            'expected_top15':   exp15,
            'fold_top15':       fold15,
            'p_top15_Fisher':   f"{p15:.2e}",
            'sig_top15':        sig(p15),
            # Top 1%
            'k_top1pct':        k1,
            'expected_top1pct': exp1,
            'p_top1pct':        f"{p1:.2e}",
            'sig_top1pct':      sig(p1),
            # Top 5%
            'k_top5pct':        k5,
            'expected_top5pct': exp5,
            'p_top5pct':        f"{p5:.2e}",
            'sig_top5pct':      sig(p5),
        })

    table_df = pd.DataFrame(rows)
    out = os.path.join(BENCHMARK_DIR, 'hypergeometric_table.csv')
    table_df.to_csv(out, index=False)

    print(f"\n  ┌─ ENRIQUECIMIENTO DE ENZIMAS HIDROLÍTICAS")
    print(f"  │  {'Organismo':<40} "
          f"{'Top15 k/exp/fold/p':^28}  "
          f"{'Top1% k/p':^18}  "
          f"{'Top5% k/p':^18}")
    print(f"  │  {'─'*105}")
    for _, r in table_df.iterrows():
        print(f"  │  {r['Organism']:<40} "
              f"k={r['k_top15']} exp={r['expected_top15']} "
              f"({r['fold_top15']}x) {r['p_top15_Fisher']:>9} {r['sig_top15']}  "
              f"k={r['k_top1pct']} {r['p_top1pct']:>9} {r['sig_top1pct']}  "
              f"k={r['k_top5pct']} {r['p_top5pct']:>9} {r['sig_top5pct']}")
    print(f"  │")
    print(f"  │  Sig: ✅ p<0.05  ~ p<0.10  ❌ p≥0.10")
    print(f"  Tabla → {out}\n")

    return table_df



# ══════════════════════════════════════════════════════════════════════════════
# FIG G — Leave-One-Organism-Out Cross Validation
# ══════════════════════════════════════════════════════════════════════════════
def fig_looo_cv(feat_cols):
    """
    Entrena y evalúa el modelo dejando fuera un organismo completo a la vez.
    Usa la columna Organism_Source del CSV de features como agrupador.
    Reporta AUC por organismo y AUC promedio ± std.
    """
    print("  Generando Fig G — LOOO-CV (puede tardar 20-40 min)...")

    df = pd.read_csv(DATA_FILE)

    # Verificar que existe columna de organismo
    if 'Organism_Source' not in df.columns:
        print("     ⚠️  Columna Organism_Source no encontrada.")
        print("        Verifica que 03_feature_extraction.py generó esta columna.")
        return None

    X   = df[feat_cols].astype(np.float32)
    y   = df['Thermal_Class'].values
    grp = df['Organism_Source'].values

    organisms = np.unique(grp)
    n_orgs    = len(organisms)
    print(f"     Organismos únicos: {n_orgs} | Proteínas: {len(df):,}")

    logo    = LeaveOneGroupOut()
    results = []

    best_params = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl")).get_params()

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, grp)):
        left_out = grp[test_idx][0]
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Saltar si el fold de test tiene una sola clase
        if len(np.unique(y_te)) < 2:
            print(f"     [{fold_i+1:02d}/{n_orgs}] {left_out:<40} — omitido (una sola clase)")
            continue

        model_fold = lgb.LGBMClassifier(**best_params)
        model_fold.fit(X_tr, y_tr)
        probs = model_fold.predict_proba(X_te.astype(np.float32))[:, 0]

        try:
            auc = roc_auc_score(1 - y_te, probs)
        except Exception:
            auc = float('nan')

        thermal = "❄️" if y_te[0] == 0 else "🌱"
        print(f"     [{fold_i+1:02d}/{n_orgs}] {left_out:<40} AUC={auc:.4f} {thermal}")
        results.append({'Organism': left_out,
                        'Thermal_Class': int(y_te[0]),
                        'N_proteins': len(test_idx),
                        'AUC': auc})

    if not results:
        print("     ❌ Sin resultados válidos.")
        return None

    res_df  = pd.DataFrame(results).dropna(subset=['AUC'])
    mean_auc = res_df['AUC'].mean()
    std_auc  = res_df['AUC'].std()
    min_auc  = res_df['AUC'].min()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(8, len(res_df) * 0.38)))

    # LOOO-EC: cada fold contiene ambas clases, color único + AUC anotado
    colors = [C_COLD] * len(res_df)
    bars = ax.barh(range(len(res_df)), res_df['AUC'].values,
                   color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
    for i, (_, row) in enumerate(res_df.iterrows()):
        ax.text(row['AUC'] + 0.005, i, f"{row['AUC']:.4f}",
                va='center', fontsize=9, color='#2c3e50')
    ax.axvline(mean_auc, color='#2c3e50', lw=2, linestyle='--',
               label=f'Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}')
    ax.axvline(0.85, color='#e74c3c', lw=1.5, linestyle=':',
               label='Publication threshold (0.85)')
    ax.axvline(0.5,  color='#bdc3c7', lw=1,   linestyle='-',
               label='Random (0.50)')

    ax.set_yticks(range(len(res_df)))
    ax.set_yticklabels(res_df['Organism'].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('AUC-ROC (leave-one-organism-out)', fontsize=10)
    ax.set_title('Leave-One-Organism-Out Cross-Validation\n'
                 'Each bar = model trained without that organism, tested on it',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_facecolor(GREY)
    ax.grid(axis='x', alpha=0.4, color='white')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc='lower right', fontsize=8.5)
    ax.set_title('Leave-One-EC-Out Cross-Validation\n'
                 'Each bar = model trained on 4 EC classes, tested on the left-out class',
                 fontsize=11, fontweight='bold')

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, '11G_LOOO_CV.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_out = os.path.join(BENCHMARK_DIR, 'looo_cv_results.csv')
    res_df.to_csv(csv_out, index=False)

    print(f"\n     ✅ LOOO-CV completado:")
    print(f"        AUC medio : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"        AUC mínimo: {min_auc:.4f}")
    print(f"        Organismos evaluados: {len(res_df)}/{n_orgs}")
    print(f"        → {out}")
    print(f"        → {csv_out}")

    return res_df, mean_auc, std_auc

# ══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY — Contribución incremental de los índices termoadaptativos
# ══════════════════════════════════════════════════════════════════════════════
def ablation_thermoadaptive():
    """
    Entrena tres versiones del modelo y compara AUC:
      1. Modelo completo (431 features)
      2. Sin los 3 índices termoadaptativos (428 features)
      3. Solo AAC + physicochemical, sin DPC ni índices (27 features)

    Si el AUC del modelo completo > modelo sin índices, los índices
    aportan información incremental más allá de los AAC individuales.
    """
    import lightgbm as lgb
    print("  Ablation study — índices termoadaptativos...")

    df       = pd.read_csv(DATA_FILE)
    fcols    = open(os.path.join(MODELS_DIR, "feature_columns.txt")).read().strip().split('\n')
    model    = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    params   = {k: v for k, v in model.get_params().items()
                if k not in ('n_jobs', 'random_state', 'verbose')}
    params.update({'n_jobs': 2, 'random_state': 42, 'verbose': -1})

    THERMO = ['IVYWREL_Index', 'CvP_Bias', 'Flexibility_Ratio']
    DPC    = [c for c in fcols if c.startswith('DPC_')]
    AAC    = [c for c in fcols if c.startswith('AAC_')]
    PHYSIO = ['Length', 'Molecular_Weight', 'GRAVY', 'Instability_Index',
              'Aromaticity', 'Helix_Fraction', 'Turn_Fraction', 'Sheet_Fraction']

    feature_sets = {
        'Full (431)':            fcols,
        'Without thermo (428)':  [c for c in fcols if c not in THERMO],
        'AAC + physico only (27)': [c for c in fcols if c in AAC + PHYSIO],
    }

    X_all = df[fcols].astype(np.float32)
    y     = df['Thermal_Class'].values

    X_tr_all, X_te_all, y_tr, y_te = train_test_split(
        X_all, y, test_size=0.20, random_state=42, stratify=y)

    results = {}
    for name, cols in feature_sets.items():
        cols_present = [c for c in cols if c in df.columns]
        X_tr = X_tr_all[[c for c in fcols if c in cols_present]]
        X_te = X_te_all[[c for c in fcols if c in cols_present]]

        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_te.astype(np.float32))[:, 0]
        auc   = roc_auc_score(1 - y_te, probs)
        results[name] = auc
        print(f"     {name:<30} AUC = {auc:.6f}")

    # Delta
    delta_thermo = results['Full (431)'] - results['Without thermo (428)']
    delta_dpc    = results['Without thermo (428)'] - results['AAC + physico only (27)']
    print(f"\n     Δ AUC por índices termoadaptativos : {delta_thermo:+.6f}")
    print(f"     Δ AUC por dipéptidos               : {delta_dpc:+.6f}")

    if abs(delta_thermo) < 0.0001:
        print(f"\n     → Índices termoadaptativos no aportan AUC incremental detectable.")
        print(f"       Argumento para el paper: su valor es interpretativo (validación")
        print(f"       biológica del modelo), no predictivo incremental.")
    else:
        print(f"\n     → Índices termoadaptativos aportan Δ AUC = {delta_thermo:+.6f}.")
        print(f"       Argumento para el paper: contribución incremental demostrada.")

    # Guardar
    abl_df = pd.DataFrame([
        {'Feature set': k, 'N features': len(feature_sets[k]), 'AUC': v}
        for k, v in results.items()
    ])
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
    parser.add_argument('--looo', action='store_true',
                        help='Ejecutar LOOO-CV (lento, ~30 min)')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Omitir comparación de baselines (rápido)')
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  PsychroScan — Figuras para Paper (11)")
    print("=" * 65 + "\n")

    model, feat_cols, X, y = load_model_and_data()

    # ── Figuras siempre ────────────────────────────────────────────────────
    feat_df  = fig_feature_importance(model, feat_cols)

    if not args.skip_baseline:
        auc_dict = fig_baseline_comparison(model, feat_cols, X, y)
        ablation_thermoadaptive()
    else:
        print("  Fig F omitida (--skip-baseline)")
        auc_dict = None

    hyper_df = table_hypergeometric()

    # ── LOOO-CV opcional ───────────────────────────────────────────────────
    looo_result = None
    if args.looo:
        looo_result = fig_looo_cv(feat_cols)
    else:
        print("  Fig G (LOOO-CV) omitida — usa --looo para ejecutarla")
        print("  Tiempo estimado en M1: 20-40 min dependiendo del dataset\n")

    # ── Resumen ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESUMEN PARA EL PAPER")
    print("=" * 65)

    if auc_dict:
        print(f"  LightGBM AUC      : {auc_dict['LightGBM']:.4f}")
        print(f"  LogReg AUC        : {auc_dict['LogisticRegression']:.4f}")
        if auc_dict['GRAVY']:
            delta_lr    = auc_dict['LightGBM'] - auc_dict['LogisticRegression']
            delta_gravy = auc_dict['LightGBM'] - auc_dict['GRAVY']
            print(f"  GRAVY AUC         : {auc_dict['GRAVY']:.4f}")
            print(f"  Δ vs LogReg       : +{delta_lr:.4f}")
            print(f"  Δ vs GRAVY        : +{delta_gravy:.4f}")

    top3_feats = feat_df.head(3)['Feature'].tolist()
    thermo_in_top30 = feat_df[feat_df['Category'] == 'Thermoadaptive']
    print(f"\n  Top 3 features    : {', '.join(top3_feats)}")
    if len(thermo_in_top30) > 0:
        names = thermo_in_top30['Feature'].tolist()
        ranks = [feat_df.index.get_loc(i) + 1
                 for i in thermo_in_top30.index]
        print(f"  Thermoadaptive en top 30: {list(zip(names, ranks))}")

    if looo_result is not None:
        _, mean_auc, std_auc = looo_result
        print(f"\n  LOOO-CV AUC       : {mean_auc:.4f} ± {std_auc:.4f}")
        if mean_auc >= 0.85:
            print(f"  ✅ Generalización validada (≥ 0.85)")
        else:
            print(f"  ⚠️  AUC < 0.85 — revisar dataset antes de publicar")

    print()
    print("  Figuras generadas en results/figures/")
    print("  Datos en results/benchmark/")
    print()




if __name__ == "__main__":
    main()