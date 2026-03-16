"""
PsychroScan — benchmark_known_enzymes.py  (v3)
===============================================
Validación retrospectiva usando el FASTA real del organismo.

En lugar de asumir accesiones externas, este script:
1. Lee el FASTA y extrae todas las proteínas con función hidrolítica anotada
   en el header (lipasas, amilasas, xilanasas, celulasas, proteasas)
2. Corre el modelo sobre el proteoma completo
3. Reporta el rank de cada enzima de interés dentro del proteoma

Esto es válido para CUALQUIER organismo — no requiere accesiones conocidas.
Solo requiere que las proteínas de interés estén anotadas en el header FASTA.

Uso:
    python benchmark_known_enzymes.py
    python benchmark_known_enzymes.py --top 50   # cambiar cutoff de reporte
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from itertools import product as iproduct

warnings.filterwarnings("ignore")

# ─── RUTAS ────────────────────────────────────────────────────────────────────
MODELS_DIR    = os.path.join("results", "models")
GENOMES_DIR   = os.path.join("data", "new_genomes")
BENCHMARK_DIR = os.path.join("results", "benchmark")
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# ─── KEYWORDS PARA DETECTAR ENZIMAS DE INTERÉS EN HEADERS FASTA ──────────────
# Cada categoría define un grupo de enzimas a seguir en el ranking.
# El script buscará estas palabras en el nombre de la proteína (case-insensitive).
# Keywords diseñados para minimizar falsos positivos.
# Regla: solo términos que aparecen EXCLUSIVAMENTE en enzimas de esa clase.
# Se excluyen siglas cortas (GH, EG, CBH) que aparecen en otros contextos.
ENZYME_KEYWORDS = {
    "Lipase / Esterase": [
        "lipase", "triacylglycerol lipase", "GDSL lipase", "cutinase",
        "phospholipase", "lysophospholipase", "monoacylglycerol lipase",
    ],
    "Esterase": [
        # Separada de lipasa para evitar contaminar con phosphodiesterase
        # Solo esterasas con nombre completo
        "carboxylesterase", "acetylesterase", "feruloyl esterase",
        "acetylcholinesterase", "tributyrinase",
    ],
    "Alpha-Amylase": [
        "alpha-amylase", "alpha amylase", "glucoamylase",
        "maltogenic amylase", "neopullulanase",
    ],
    "Cellulase": [
        "cellulase", "endoglucanase", "cellobiohydrolase",
        "endo-1,4-beta-glucanase", "exoglucanase",
    ],
    "Xylanase / Hemicellulase": [
        "xylanase", "endo-1,4-beta-xylanase", "beta-xylosidase",
        "xylosidase", "arabinoxylanase", "mannanase", "arabinofuranosidase",
    ],
    "Protease": [
        "protease", "peptidase", "subtilisin", "aminopeptidase",
        "metalloprotease", "serine protease", "endopeptidase",
        "carboxypeptidase", "aspartyl protease",
    ],
    "Glucosidase / Galactosidase": [
        "glucosidase", "galactosidase", "beta-glucosidase",
        "alpha-glucosidase", "mannosidase", "beta-galactosidase",
    ],
}

# ─── CONSTANTES DE FEATURES ───────────────────────────────────────────────────
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES  = [''.join(p) for p in iproduct(AMINO_ACIDS, repeat=2)]
VALID_AA    = set(AMINO_ACIDS)
IVYWREL_SET = set("IVYWREL")
CHARGED_SET = set("RKDE")
POLAR_SET   = set("NQST")


def load_model():
    model  = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    thresh = float(open(os.path.join(MODELS_DIR, "threshold.txt")).read().strip())
    fcols  = open(os.path.join(MODELS_DIR, "feature_columns.txt")).read().strip().split('\n')
    return model, thresh, fcols


def parse_accession(header):
    """sp|B6HAA7|ARO1_PENRW → B6HAA7"""
    parts = header.split('|')
    return parts[1] if len(parts) >= 2 else header.split()[0]


def parse_description(header):
    """Extrae la descripción funcional del header FASTA."""
    # "sp|B6HAA7|ARO1_PENRW Pentafunctional AROM... OS=..." → "Pentafunctional AROM..."
    m = re.search(r'\|[^|]+\s+(.*?)\s+OS=', header)
    if m:
        return m.group(1)
    # Fallback: todo después del primer espacio
    parts = header.split(' ', 1)
    return parts[1] if len(parts) > 1 else header


def classify_enzyme(description):
    """Clasifica una proteína según su descripción. Devuelve categoría o None."""
    desc_lower = description.lower()
    for category, keywords in ENZYME_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in desc_lower:
                return category
    return None


def extract_features(fasta_path):
    records = []
    for rec in SeqIO.parse(fasta_path, 'fasta'):
        seq = str(rec.seq).upper()
        if not set(seq).issubset(VALID_AA) or len(seq) < 20:
            continue
        try:
            analysis = ProteinAnalysis(seq)
            ss   = analysis.secondary_structure_fraction()
            aac  = analysis.amino_acids_percent
            slen = len(seq)
            feat = {
                'Protein_ID':   rec.id,
                'Description':  parse_description(rec.description),
                'Accession':    parse_accession(rec.id),
                'Length':       slen,
                'Molecular_Weight':  analysis.molecular_weight(),
                'GRAVY':             analysis.gravy(),
                'Instability_Index': analysis.instability_index(),
                'Aromaticity':       analysis.aromaticity(),
                'Helix_Fraction':    ss[0],
                'Turn_Fraction':     ss[1],
                'Sheet_Fraction':    ss[2],
                'IVYWREL_Index':    sum(1 for a in seq if a in IVYWREL_SET) / slen,
                'CvP_Bias':         (sum(1 for a in seq if a in CHARGED_SET) -
                                     sum(1 for a in seq if a in POLAR_SET)) / slen,
                'Flexibility_Ratio':(seq.count('G') + seq.count('S')) /
                                    (seq.count('P') + 0.001),
            }
            for aa in AMINO_ACIDS:
                feat[f'AAC_{aa}'] = aac[aa]
            for di in DIPEPTIDES:
                feat[f'DPC_{di}'] = seq.count(di) / (slen - 1)
            records.append(feat)
        except Exception:
            continue
    return pd.DataFrame(records)


def run_benchmark(top_n=15):
    print("\n" + "=" * 68)
    print("  PsychroScan — Benchmark de Validación Retrospectiva (v3)")
    print("=" * 68)

    fastas = sorted(f for f in os.listdir(GENOMES_DIR) if f.endswith('.fasta'))
    if not fastas:
        print(f"  No hay .fasta en {GENOMES_DIR}/")
        sys.exit(0)

    model, threshold, feat_cols = load_model()

    for fasta_file in fastas:
        organism  = fasta_file.replace('.fasta', '')
        fasta_path = os.path.join(GENOMES_DIR, fasta_file)

        print(f"\n  Organismo : {organism}")
        print(f"  Extrayendo features...", end=" ", flush=True)
        df = extract_features(fasta_path)
        if len(df) == 0:
            print("sin proteínas válidas.")
            continue
        print(f"{len(df):,} proteínas.")

        # ── Predicción ────────────────────────────────────────────────────────
        X = df[[c for c in feat_cols if c in df.columns]].copy()
        for c in feat_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feat_cols].astype(np.float32)

        probs = model.predict_proba(X)[:, 0]
        df['Cold_Probability'] = probs
        df['Enzyme_Category']  = df['Description'].apply(classify_enzyme)
        df_ranked = df.sort_values('Cold_Probability', ascending=False).reset_index(drop=True)
        df_ranked['Rank']      = df_ranked.index + 1
        df_ranked['Percentile']= (1 - df_ranked['Rank'] / len(df_ranked)) * 100

        n_total   = len(df_ranked)
        enzymes_df = df_ranked[df_ranked['Enzyme_Category'].notna()].copy()

        # ── 1. Top N del proteoma completo ────────────────────────────────────
        print(f"\n  ┌─ TOP {top_n} DEL PROTEOMA COMPLETO")
        print(f"  │  {'#':<4} {'Accesión':<12} {'P(Cold)':>8}  Descripción")
        print(f"  │  {'─'*60}")
        for _, row in df_ranked.head(top_n).iterrows():
            cat_tag = f" [{row['Enzyme_Category']}]" if pd.notna(row['Enzyme_Category']) else ""
            desc    = row['Description'][:45]
            print(f"  │  {int(row['Rank']):<4} {row['Accession']:<12} "
                  f"{row['Cold_Probability']*100:>7.2f}%  {desc}{cat_tag}")

        # ── 2. Enzimas hidrolíticas detectadas y su rank ──────────────────────
        print(f"\n  ┌─ ENZIMAS HIDROLÍTICAS: RANK EN EL PROTEOMA")
        print(f"  │  {len(enzymes_df):,} proteínas con función hidrolítica anotada "
              f"de {n_total:,} totales\n")

        color_map = {
            "✅ Top 1%":   lambda p: p >= 99,
            "✅ Top 5%":   lambda p: p >= 95,
            "🟡 Top 10%":  lambda p: p >= 90,
            "🟠 Top 25%":  lambda p: p >= 75,
        }

        for category in ENZYME_KEYWORDS:
            cat_df = enzymes_df[enzymes_df['Enzyme_Category'] == category]
            if len(cat_df) == 0:
                continue
            print(f"  │  ── {category} ({len(cat_df)} proteínas)")
            print(f"  │     {'Rank':>6}  {'Top%':>6}  {'P(Cold)':>8}  Accesión      Descripción")
            for _, row in cat_df.head(8).iterrows():
                pct = row['Percentile']
                status = ("✅" if pct >= 95 else "🟡" if pct >= 75 else "🔴")
                print(f"  │     {int(row['Rank']):>6}  {pct:>5.1f}%  "
                      f"{row['Cold_Probability']*100:>7.2f}%  "
                      f"{row['Accession']:<12}  {row['Description'][:40]}")
            if len(cat_df) > 8:
                print(f"  │     ... y {len(cat_df)-8} más")
            print(f"  │")

        # ── 3. Estadísticas de recuperación ───────────────────────────────────
        n_in_top15  = (enzymes_df['Rank'] <= 15).sum()
        n_in_top100 = (enzymes_df['Rank'] <= 100).sum()
        n_in_top5pct= (enzymes_df['Percentile'] >= 95).sum()

        print(f"\n  ┌─ RESUMEN DE RECUPERACIÓN")
        print(f"  │  Enzimas en Top 15          : {n_in_top15} / {len(enzymes_df)}")
        print(f"  │  Enzimas en Top 100         : {n_in_top100} / {len(enzymes_df)}")
        print(f"  │  Enzimas en Top 5% proteoma : {n_in_top5pct} / {len(enzymes_df)}")

        if n_in_top15 >= 1:
            # Mostrar cuáles son
            top15_enzymes = enzymes_df[enzymes_df['Rank'] <= 15]
            print(f"\n  ⭐ VALIDACIÓN POSITIVA: {n_in_top15} enzima(s) de interés en Top 15:")
            for _, row in top15_enzymes.iterrows():
                print(f"     Rank {int(row['Rank'])}: {row['Accession']} — "
                      f"{row['Description'][:50]} [{row['Enzyme_Category']}]")
            print(f"     → Resultado citable para el paper.")
        elif n_in_top5pct >= 1:
            print(f"\n  🟡 VALIDACIÓN PARCIAL: enzimas de interés priorizadas")
            print(f"     en Top 5% pero no en Top 15.")
            print(f"     → Considera ampliar el output del script 09 a Top 50.")
        else:
            print(f"\n  🔴 MODELO NO PRIORIZA enzimas hidrolíticas de este organismo.")
            print(f"     Verifica que el organismo sea psicrotrófico/psicrófilo.")

        # ── 4. Guardar ────────────────────────────────────────────────────────
        out_path = os.path.join(BENCHMARK_DIR, f"{organism}_benchmark.csv")
        df_ranked.to_csv(out_path, index=False)
        print(f"\n  Ranking completo guardado → {out_path}")

    print(f"\n{'='*68}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=15,
                        help='Cuántas proteínas mostrar en el Top N (default: 15)')
    args = parser.parse_args()
    run_benchmark(top_n=args.top)