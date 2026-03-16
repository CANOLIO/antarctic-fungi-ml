import os
import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
NEW_GENOMES_DIR = os.path.join("data", "new_genomes")
MODELS_DIR      = os.path.join("results", "models")
DATA_FILE       = os.path.join("data", "processed", "dataset_features.csv")
os.makedirs(NEW_GENOMES_DIR, exist_ok=True)

# ─── CONSTANTES ───────────────────────────────────────────────────────────────
AMINO_ACIDS  = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES   = [''.join(p) for p in product(AMINO_ACIDS, repeat=2)]
VALID_AA     = set(AMINO_ACIDS)
META_COLS    = ['Protein_ID', 'Organism_Source', 'EC_Class', 'Thermal_Class']
IVYWREL_SET  = set("IVYWREL")
CHARGED_SET  = set("RKDE")
POLAR_SET    = set("NQST")

INDUSTRIAL_PFAM = {
    'Lipase_GDSL', 'LipaseX', 'Lipase', 'Abhydrolase_1', 'Abhydrolase_3',
    'Trypsin', 'Peptidase_S8', 'Peptidase_M4', 'Peptidase_M10',
    'Alpha-amylase', 'Glyco_hydro_13', 'Glyco_hydro_5', 'Cellulase',
    'CBM_1', 'GH18', 'PL_1',
}


def load_model_config():
    model     = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    thresh    = float(open(os.path.join(MODELS_DIR, "threshold.txt")).read().strip()) \
                if os.path.exists(os.path.join(MODELS_DIR, "threshold.txt")) else 0.5
    feat_path = os.path.join(MODELS_DIR, "feature_columns.txt")
    feat_cols = open(feat_path).read().strip().split('\n') \
                if os.path.exists(feat_path) else None
    return model, thresh, feat_cols


def extract_features(filepath):
    features_list, prot_ids = [], []
    print(f"  Extrayendo features de {os.path.basename(filepath)}...")
    for record in SeqIO.parse(filepath, 'fasta'):
        seq = str(record.seq).upper()
        if not set(seq).issubset(VALID_AA) or len(seq) < 20:
            continue
        try:
            analysis = ProteinAnalysis(seq)
            ss, aac  = analysis.secondary_structure_fraction(), analysis.amino_acids_percent
            slen     = len(seq)
            feat = {
                'Length': slen,
                'Molecular_Weight': analysis.molecular_weight(),
                'GRAVY': analysis.gravy(),
                'Instability_Index': analysis.instability_index(),
                'Aromaticity': analysis.aromaticity(),
                'Helix_Fraction': ss[0], 'Turn_Fraction': ss[1], 'Sheet_Fraction': ss[2],
                # Features termoadaptativos v3
                'IVYWREL_Index':    sum(1 for a in seq if a in IVYWREL_SET) / slen,
                'CvP_Bias':         (sum(1 for a in seq if a in CHARGED_SET) -
                                     sum(1 for a in seq if a in POLAR_SET)) / slen,
                'Flexibility_Ratio':(seq.count('G') + seq.count('S')) / (seq.count('P') + 0.001),
            }
            for aa in AMINO_ACIDS:
                feat[f'AAC_{aa}'] = aac[aa]
            for di in DIPEPTIDES:
                feat[f'DPC_{di}'] = seq.count(di) / (slen - 1)
            features_list.append(feat)
            prot_ids.append(record.id)
        except Exception:
            continue
    return pd.DataFrame(features_list), prot_ids


def compute_industrial_score(top100_ids):
    session = requests.Session()
    retry   = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    industrial_count = 0
    print(f"  Consultando Pfam Top-100 (industrial_score)...", end=" ", flush=True)
    for prot_id in top100_ids[:100]:
        try:
            acc = prot_id.split('|')[1] if '|' in prot_id else prot_id.split()[0]
            r   = session.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json", timeout=15)
            if r.status_code != 200:
                continue
            for xref in r.json().get('crossReferences', []):
                if xref.get('database') == 'Pfam':
                    name = next((p['value'] for p in xref.get('properties', [])
                                 if p['key'] == 'EntryName'), '')
                    if any(ind in name for ind in INDUSTRIAL_PFAM):
                        industrial_count += 1
                        break
        except Exception:
            continue
        time.sleep(0.3)
    score = industrial_count / min(len(top100_ids), 100)
    print(f"{industrial_count}/{min(len(top100_ids),100)} → {score:.3f}")
    return score


def compute_ppi_v2(df_results, X_feat, query_pfam=True):
    top100       = df_results.nlargest(100, 'Cold_Probability')
    top100_mean  = top100['Cold_Probability'].mean()
    pct_above_90 = (df_results['Cold_Probability'] >= 0.90).mean()
    flex_raw     = (X_feat['AAC_G'].mean() + X_feat['AAC_S'].mean()) / (X_feat['AAC_P'].mean() + 0.001)
    norm_flex    = min(flex_raw / 8.0, 1.0)
    ind_score    = compute_industrial_score(top100['Protein_ID'].tolist()) if query_pfam else 0.0
    ppi = (0.40 * top100_mean + 0.25 * pct_above_90 + 0.15 * norm_flex + 0.20 * ind_score) * 100
    return {'ppi': ppi, 'top100_mean': top100_mean, 'pct_above_90': pct_above_90,
            'flex_raw': flex_raw, 'industrial_score': ind_score}


# ─── PANEL DE REFERENCIA ──────────────────────────────────────────────────────
PANEL_FILE = os.path.join("data", "reference_panel", "reference_profiles.csv")


def load_reference_panel():
    """
    Carga el panel de referencia taxonómica construido por
    10_build_reference_panel.py. Si no existe, avisa al usuario.
    """
    if not os.path.exists(PANEL_FILE):
        print(f"  ⚠️  Panel de referencia no encontrado: {PANEL_FILE}")
        print(f"     Ejecuta primero: python 10_build_reference_panel.py")
        return None
    df = pd.read_csv(PANEL_FILE)
    print(f"  Panel de referencia: {len(df)} organismos cargados.")
    return df


def build_context_dendrogram(X_feat, organism_name, panel_df, output_path):
    """
    Dendrograma de posicionamiento proteómico v2.
    Usa el panel de referencia taxonómica (organismos reales con identidad
    conocida) en lugar de promedios por clase EC.

    Colores:
      - Azul  (#4a90d9): psicrófilo / psicrotrofo (Thermal_Class = 0)
      - Naranja (#e07b54): mesófilo (Thermal_Class = 1)
      - Verde: candidato nuevo
    """
    from matplotlib.lines import Line2D

    if panel_df is None or len(panel_df) == 0:
        print("  ⚠️  Sin panel de referencia — dendrograma omitido.")
        return

    meta_panel = ['Organism_Name', 'Organism_Label', 'Thermal_Class', 'N_sequences']
    feat_cols  = [c for c in panel_df.columns if c not in meta_panel]

    # Alinear columnas: el panel puede tener features que X_feat no tiene y viceversa
    common_feats = [c for c in feat_cols if c in X_feat.columns]
    if len(common_feats) < 10:
        print(f"  ⚠️  Solo {len(common_feats)} features en común — dendrograma omitido.")
        return

    # Perfiles de referencia
    ref_profiles     = panel_df[common_feats].values
    ref_labels       = panel_df['Organism_Label'].tolist()
    ref_thermal      = panel_df['Thermal_Class'].tolist()

    # Perfil del candidato
    new_label        = f"★ {organism_name.upper()}"
    new_profile      = X_feat[common_feats].mean().values.reshape(1, -1)

    combined_data    = np.vstack([ref_profiles, new_profile])
    combined_labels  = ref_labels + [new_label]
    combined_thermal = ref_thermal + [-1]   # -1 = candidato nuevo

    # Dendrograma
    scaler = StandardScaler()
    Z      = hierarchy.linkage(scaler.fit_transform(combined_data), method='ward')

    n_orgs = len(combined_labels)
    fig, ax = plt.subplots(figsize=(13, max(12, n_orgs * 0.45)))
    ax.set_title(
        f"Proteome-level Positioning: {organism_name.upper()} vs Reference Panel",
        fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Biochemical Distance (Adaptive Convergence)", fontsize=10)

    hierarchy.dendrogram(
        Z, labels=combined_labels, orientation='left',
        leaf_font_size=9, ax=ax, color_threshold=0,
        above_threshold_color='#aaaaaa',
    )

    # Colorear etiquetas
    color_map = {-1: '#2ecc71', 0: '#4a90d9', 1: '#e07b54'}
    thermal_by_label = dict(zip(combined_labels, combined_thermal))
    for tick in ax.get_ymajorticklabels():
        lbl    = tick.get_text()
        tc     = thermal_by_label.get(lbl, 1)
        tick.set_color(color_map[tc])
        if tc == -1:
            tick.set_fontweight('bold')
            tick.set_fontsize(10)

    ax.legend(handles=[
        Line2D([0],[0], color='#4a90d9', lw=3, label='Psychrophile / psychrotroph'),
        Line2D([0],[0], color='#e07b54', lw=3, label='Mesophile'),
        Line2D([0],[0], color='#2ecc71', lw=3, label=f'Candidate: {organism_name}'),
    ], loc='lower right', fontsize=9, framealpha=0.9)

    ax.text(0.01, 0.01,
        f"n={len(common_feats)} features  ·  {len(ref_labels)} organismos de referencia  ·  método: Ward",
        transform=ax.transAxes, fontsize=7, color='#888888', va='bottom')

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Dendrograma → {output_path}")


def analyze(query_pfam=True):
    print("\n" + "=" * 70)
    print("  PsychroScan — Motor de Predicción (PPI v2.0)")
    print("=" * 70 + "\n")

    fastas = [f for f in os.listdir(NEW_GENOMES_DIR) if f.endswith('.fasta')]
    if not fastas:
        print(f"  No hay .fasta en {NEW_GENOMES_DIR}/")
        sys.exit(0)

    model, threshold, feat_cols = load_model_config()
    original_df  = pd.read_csv(DATA_FILE)
    reference_df = load_reference_panel()
    print(f"  Dataset entrenamiento : {len(original_df):,} proteínas")
    print(f"  Umbral de decisión    : {threshold:.4f}\n")

    for fasta_file in sorted(fastas):
        organism_name   = fasta_file.replace('.fasta', '')
        X_new, prot_ids = extract_features(os.path.join(NEW_GENOMES_DIR, fasta_file))

        if len(X_new) == 0:
            print(f"  Sin proteínas válidas en {fasta_file}.")
            continue

        if feat_cols:
            for c in feat_cols:
                if c not in X_new.columns:
                    X_new[c] = 0.0
            X_new = X_new[feat_cols]

        probs_cold = model.predict_proba(X_new.astype(np.float32))[:, 0]
        df_res     = pd.DataFrame({'Protein_ID': prot_ids, 'Cold_Probability': probs_cold})

        ppi_data = compute_ppi_v2(df_res, X_new, query_pfam=query_pfam)
        ppi      = ppi_data['ppi']
        veredicto = ("❄️  EXTREMÓFILO FRÍO" if ppi > 50 else
                     "🌥️  TOLERANTE AL FRÍO" if ppi > 30 else "🌱  MESÓFILO")

        print(f"\n  {'='*60}")
        print(f"  REPORTE: {organism_name.upper()}")
        print(f"  {'='*60}")
        print(f"  Veredicto      : {veredicto}")
        print(f"  PPI v2.0       : {ppi:.2f} / 100")
        print(f"    Media Top-100: {ppi_data['top100_mean']*100:.1f}%  (40%)")
        print(f"    Frac > 90%   : {ppi_data['pct_above_90']*100:.2f}% (25%)")
        print(f"    G+S/Pro      : {ppi_data['flex_raw']:.2f}         (15%)")
        print(f"    Ind. score   : {ppi_data['industrial_score']:.3f}       (20%)")

        print(f"\n  TOP 5 CANDIDATAS:")
        print(f"  {'-'*58}")
        for _, row in df_res.nlargest(5, 'Cold_Probability').iterrows():
            idx   = prot_ids.index(row['Protein_ID'])
            gly   = X_new.iloc[idx]['AAC_G']
            pro   = X_new.iloc[idx]['AAC_P']
            helix = X_new.iloc[idx]['Helix_Fraction'] * 100
            print(f"  {row['Protein_ID'][:30]:<30} P={row['Cold_Probability']*100:.1f}%")
            print(f"    Gly={gly:.1f}%  Pro={pro:.1f}%  Helice={helix:.1f}%")

        dend_path = os.path.join(NEW_GENOMES_DIR, f"{organism_name}_PPI_context.png")
        build_context_dendrogram(X_new, organism_name, reference_df, dend_path)

        df_res.to_csv(
            os.path.join(NEW_GENOMES_DIR, f"{organism_name}_full_results.csv"), index=False)
        print(f"  Resultados → {organism_name}_full_results.csv\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-pfam', action='store_true')
    args = parser.parse_args()
    analyze(query_pfam=not args.no_pfam)