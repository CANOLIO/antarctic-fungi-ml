"""
PsychroScan — 08_publishable_validations.py  (v2)
==================================================
Genera las figuras de validación científica publicables.

CAMBIOS RESPECTO A v1:
  1. Carga umbral desde results/models/threshold.txt (no hardcodeado).
  2. Target correcto: Thermal_Class, Cold=0 sin inversión de etiquetas.
  3. Figura D (nueva): Lollipop chart — Top 15 con P(Cold) y dominio Pfam.
  4. PCA coloreado por Thermal_Class (no por taxonomía) — ya estaba correcto.
  5. Boxplot Gly/Ser/Pro ahora incluye Pro como marcador de rigidez mesófila.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Sin display (compatible con entornos sin GUI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─── RUTAS ────────────────────────────────────────────────────────────────────
DATA_FILE   = os.path.join("data", "processed", "dataset_features.csv")
MODELS_DIR  = os.path.join("results", "models")
FIGURES_DIR = os.path.join("results", "figures")
REPORT_FILE = os.path.join("results", "top15_bioprospecting_report.csv")
os.makedirs(FIGURES_DIR, exist_ok=True)

META_COLS = ['Protein_ID', 'Organism_Source', 'EC_Class', 'Thermal_Class']

# ─── ESTILO GLOBAL ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLD_COLOR = "#4a90d9"   # Azul frío
WARM_COLOR = "#e07b54"   # Naranja cálido


def load_threshold() -> float:
    path = os.path.join(MODELS_DIR, "threshold.txt")
    if os.path.exists(path):
        with open(path) as f:
            return float(f.read().strip())
    print("  ⚠️  threshold.txt no encontrado. Usando 0.5 por defecto.")
    return 0.5


def generate_figures():
    print("\n" + "=" * 65)
    print("  PsychroScan — Figuras de Validación (v2)")
    print("=" * 65 + "\n")

    print("  Cargando datos y modelo...")
    df    = pd.read_csv(DATA_FILE)
    model = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    thresh = load_threshold()

    feat_cols     = [c for c in df.columns if c not in META_COLS]
    X             = df[feat_cols].astype('float32')
    y             = df['Thermal_Class'].values          # 0=Cold, 1=Warm

    probs_cold    = model.predict_proba(X)[:, 0]        # P(Cold)
    y_cold_binary = (1 - y)                             # 1=Cold para ROC

    # ── FIGURA A: Curva ROC ───────────────────────────────────────────────────
    print("  Generando Fig A — Curva ROC...")
    fpr, tpr, _ = roc_curve(y_cold_binary, probs_cold)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLD_COLOR, lw=2.5,
            label=f'Clasificador PsychroScan (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Aleatorio (AUC = 0.500)')
    ax.fill_between(fpr, tpr, alpha=0.08, color=COLD_COLOR)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('Tasa de Falsos Positivos (1 − Especificidad)', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12)
    ax.set_title('Figura A — Rendimiento del Clasificador de Enzimas Frías', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    if roc_auc >= 0.85:
        ax.text(0.55, 0.12, f'✓ Nivel publicable (AUC ≥ 0.85)',
                fontsize=10, color='green', transform=ax.transAxes)

    fig.savefig(os.path.join(FIGURES_DIR, '08A_ROC_Curve.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"     AUC = {roc_auc:.4f}")

    # ── FIGURA B: PCA ─────────────────────────────────────────────────────────
    print("  Generando Fig B — PCA del espacio proteómico...")
    n_sample = min(20_000, len(df))
    df_pca   = df.sample(n=n_sample, random_state=42)
    X_pca    = df_pca[feat_cols].astype('float32')
    y_pca    = df_pca['Thermal_Class'].values

    scaler = StandardScaler()
    pcs    = PCA(n_components=2).fit_transform(scaler.fit_transform(X_pca))
    pca_obj = PCA(n_components=2).fit(scaler.fit_transform(X_pca))   # para varianza

    explained = pca_obj.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = np.where(y_pca == 0, COLD_COLOR, WARM_COLOR)
    ax.scatter(pcs[:, 0], pcs[:, 1], c=colors, alpha=0.45, s=12, edgecolors='none')

    cold_patch = mpatches.Patch(color=COLD_COLOR, label='Enzima Fría (Cold, 0)')
    warm_patch = mpatches.Patch(color=WARM_COLOR, label='Enzima Mesófila (Warm, 1)')
    ax.legend(handles=[cold_patch, warm_patch], loc='upper right', fontsize=11)
    ax.set_xlabel(f'PC1 ({explained[0]:.1f}% varianza)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}% varianza)', fontsize=12)
    ax.set_title('Figura B — PCA del Espacio de Features Proteómicos\n'
                 '(Coloreado por Clase Térmica)', fontsize=13)

    fig.savefig(os.path.join(FIGURES_DIR, '08B_PCA_Space.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── FIGURA C: Boxplot Gly / Ser / Pro ─────────────────────────────────────
    print("  Generando Fig C — Firma aminoacídica (Gly, Ser, Pro)...")
    df['Cold_Prob'] = probs_cold

    top_cold = df[df['Cold_Prob'] > thresh].copy()
    top_cold['Grupo'] = 'Candidatas Frías'

    n_top = max(len(top_cold), 10)
    meso  = df[(df['Thermal_Class'] == 1) & (df['Cold_Prob'] < (1 - thresh))]
    if len(meso) > n_top:
        meso = meso.sample(n=n_top, random_state=42)
    meso = meso.copy()
    meso['Grupo'] = 'Mesófilas Base'

    cmp_df = pd.concat([top_cold, meso])
    melted = pd.melt(cmp_df, id_vars=['Grupo'],
                     value_vars=['AAC_G', 'AAC_S', 'AAC_P'],
                     var_name='Aminoácido', value_name='Fracción')
    melted['Aminoácido'] = melted['Aminoácido'].map({
        'AAC_G': 'Glicina\n(Flexibilidad)',
        'AAC_S': 'Serina\n(Puentes H)',
        'AAC_P': 'Prolina\n(Rigidez)',
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Aminoácido', y='Fracción', hue='Grupo',
                data=melted, palette={'Candidatas Frías': COLD_COLOR,
                                      'Mesófilas Base': WARM_COLOR},
                showfliers=False, ax=ax, width=0.5)
    ax.set_title('Figura C — Firma Aminoacídica de Adaptación Térmica', fontsize=13)
    ax.set_xlabel('')
    ax.set_ylabel('Fracción en la Secuencia', fontsize=12)
    ax.legend(title='Grupo', fontsize=11)

    fig.savefig(os.path.join(FIGURES_DIR, '08C_AA_Composition.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── FIGURA D: Lollipop Top 15 con Pfam ────────────────────────────────────
    print("  Generando Fig D — Top 15 candidatas con anotación Pfam...")

    if not os.path.exists(REPORT_FILE):
        print("    ⚠️  top15_bioprospecting_report.csv no encontrado.")
        print("       Corre primero 07_biological_annotation.py para generar la Fig D.")
    else:
        rep = pd.read_csv(REPORT_FILE)
        rep = rep.sort_values('P_Cold', ascending=True)

        # Etiqueta corta para el eje Y
        rep['Label'] = rep.apply(
            lambda r: f"{r['Protein_ID'][:18]}  [{r['EC_Class'][:8]}]", axis=1
        )
        rep['Has_Pfam'] = rep['Pfam_Domains'] != 'Sin dominios Pfam'
        rep['P_Cold_num'] = rep['P_Cold'].str.replace('%', '').astype(float)

        colors_d = [COLD_COLOR if hpf else '#adb5bd' for hpf in rep['Has_Pfam']]

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.hlines(y=rep['Label'], xmin=0, xmax=rep['P_Cold_num'],
                  color='#dee2e6', linewidth=2)
        ax.scatter(rep['P_Cold_num'], rep['Label'],
                   c=colors_d, s=90, zorder=5)

        ax.set_xlabel('P(Cold) — Probabilidad de Adaptación al Frío (%)', fontsize=12)
        ax.set_title('Figura D — Top 15 Candidatas Industriales\n'
                     'Azul = con dominio Pfam catalítico  |  Gris = sin Pfam', fontsize=13)
        ax.set_xlim([0, 105])
        ax.axvline(x=thresh * 100, color='red', linestyle='--',
                   linewidth=1.2, label=f'Umbral ({thresh*100:.0f}%)')
        ax.legend(fontsize=10)

        fig.savefig(os.path.join(FIGURES_DIR, '08D_Top15_Pfam.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"     {rep['Has_Pfam'].sum()}/15 proteínas con dominio Pfam graficadas.")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  FIGURAS GENERADAS")
    print("=" * 65)
    for fname in sorted(os.listdir(FIGURES_DIR)):
        if fname.startswith('08') and fname.endswith('.png'):
            fpath   = os.path.join(FIGURES_DIR, fname)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  📊  {fname:<40} {size_kb:>5.0f} KB")
    print(f"\n  ✅ Revisa results/figures/ para los gráficos.")
    print("  ✅ Siguiente paso → 09_predict_new_genome.py\n")


if __name__ == "__main__":
    generate_figures()