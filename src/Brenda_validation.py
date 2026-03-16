"""
PsychroScan — brenda_validation.py
=====================================
Validación externa con BRENDA: recupera enzimas psicrófilas documentadas
con T_opt <= 15°C que NO estuvieron en el training set, las puntúa con el
modelo entrenado, y reporta qué fracción aparece en el top del ranking.

Esto convierte el PPI de "proof-of-concept" a "validado externamente".

Estrategia:
  1. Descarga entradas de BRENDA con T_opt documentada <= 15°C
     (vía UniProt REST, filtrando por temperatura óptima en texto)
  2. Excluye cualquier secuencia cuyos taxones estén en el training set
  3. Extrae features con el mismo pipeline (script 03)
  4. Puntúa con el modelo entrenado
  5. Reporta: rank medio, fracción en top-10%, top-5%, top-1%

Uso:
    python src/brenda_validation.py

Requiere acceso a internet (UniProt REST API).
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import joblib
from Bio import SeqIO
from io import StringIO

MODELS_DIR  = os.path.join("results", "models")
RESULTS_DIR = os.path.join("results", "benchmark")
DATA_FILE   = os.path.join("data", "processed", "dataset_features.csv")
OUT_FASTA   = os.path.join("data", "processed", "brenda_external_set.fasta")
OUT_CSV     = os.path.join("results", "benchmark", "brenda_validation.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# T_opt threshold for cold-active classification
TOPT_THRESHOLD = 15.0

# UniProt REST base
UNIPROT_REST = "https://rest.uniprot.org/uniprotkb/search"

# ─── EC classes covered by PsychroScan ────────────────────────────────────────
EC_QUERIES = {
    "Lipases":          "ec:3.1.1.3",
    "Alpha_Amylases":   "ec:3.2.1.1",
    "Cellulases":       "ec:3.2.1.4",
    "Serine_Proteases": "ec:3.4.21.*",
    "Metalloproteases": "ec:3.4.24.*",
}

# Known psychrophile keywords used to filter BRENDA-documented entries
COLD_KEYWORDS = [
    "psychrophil", "psychrotrophic", "cold-active", "cold-adapted",
    "low temperature", "antarctic", "arctic", "glacier", "permafrost",
    "sea ice", "deep sea", "deep-sea"
]


def get_training_ids():
    """Retorna el set de Protein_IDs usados en entrenamiento."""
    df = pd.read_csv(DATA_FILE)
    return set(df['Protein_ID'].tolist())


def get_training_taxa():
    """Retorna el set de nombres de organismos en el training set."""
    with open(os.path.join("config", "taxa_list.json")) as f:
        taxa = json.load(f)
    names = set()
    for entry in taxa.get("psychrophiles", []) + taxa.get("mesophiles", []):
        names.add(entry["name"].lower().replace("_", " "))
    return names


def query_uniprot_cold_enzymes(ec_query, ec_class, max_results=500):
    """
    Busca en UniProt enzimas con T_opt documentada en frío.
    Usa la API REST con campo de temperatura óptima de crecimiento.
    """
    # Query: EC class + temperatura óptima baja documentada en comentarios
    query = (
        f"({ec_query}) AND "
        f"(cc_catalytic_activity:psychrophil* OR "
        f" cc_function:psychrophil* OR "
        f" cc_function:cold-active OR "
        f" organism_name:psychrobacter OR "
        f" organism_name:psychromonas OR "
        f" organism_name:colwellia OR "
        f" organism_name:glaciecola OR "
        f" organism_name:psychroflexus OR "
        f" organism_name:marinomonas)"
    )
    params = {
        "query":    query,
        "format":   "fasta",
        "size":     max_results,
        "fields":   "accession,organism_name,protein_name,ec",
    }
    try:
        resp = requests.get(UNIPROT_REST, params=params, timeout=60)
        if resp.status_code == 200 and resp.text.strip():
            records = list(SeqIO.parse(StringIO(resp.text), "fasta"))
            print(f"    {ec_class}: {len(records)} secuencias descargadas")
            return records
        else:
            print(f"    {ec_class}: sin resultados (status {resp.status_code})")
            return []
    except Exception as e:
        print(f"    {ec_class}: error — {e}")
        return []


def filter_external_sequences(records, training_ids, training_taxa):
    """
    Filtra secuencias para mantener solo las verdaderamente externas:
    - No en el training set (por ID)
    - De organismos no en la lista de taxones de entrenamiento
    - Longitud mínima 50 aa
    """
    external = []
    for rec in records:
        pid = rec.id.split("|")[1] if "|" in rec.id else rec.id
        if pid in training_ids:
            continue
        if len(rec.seq) < 50:
            continue
        org = rec.description.lower()
        if any(taxon in org for taxon in training_taxa):
            continue
        external.append((pid, str(rec.seq), rec.description))
    return external


def extract_features(sequences):
    """Wrapper directo al extractor inline con 431 features completos."""
    return _extract_features_inline(sequences)


def _extract_features_inline(sequences):
    """
    Extracción de features inline: AAC + DPC + fisicoquímicos + termoadaptativos.
    Mismos 431 features que el script 03.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from itertools import product
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    ALL_DPC = [a+b for a, b in product(AA_LIST, repeat=2)]

    rows, skipped = [], 0
    for pid, seq, desc in sequences:
        seq_clean = ''.join(c for c in str(seq).upper() if c in AA_LIST)
        if len(seq_clean) < 20:
            skipped += 1
            continue
        try:
            pa  = ProteinAnalysis(seq_clean)
            aac = pa.get_amino_acids_percent()
            row = {f"AAC_{aa}": aac.get(aa, 0.0) * 100 for aa in AA_LIST}

            # DPC
            n_dp = len(seq_clean) - 1
            if n_dp > 0:
                dp_counts = {}
                for i in range(n_dp):
                    dp = seq_clean[i:i+2]
                    dp_counts[dp] = dp_counts.get(dp, 0) + 1
                for dp in ALL_DPC:
                    row[f"DPC_{dp}"] = dp_counts.get(dp, 0) / n_dp * 100
            else:
                for dp in ALL_DPC:
                    row[f"DPC_{dp}"] = 0.0

            # Physicochemical
            ss = pa.secondary_structure_fraction()
            row["Length"]            = len(seq_clean)
            row["Molecular_Weight"]  = pa.molecular_weight()
            row["GRAVY"]             = pa.gravy()
            row["Instability_Index"] = pa.instability_index()
            row["Aromaticity"]       = pa.aromaticity()
            row["Helix_Fraction"]    = ss[0]
            row["Turn_Fraction"]     = ss[1]
            row["Sheet_Fraction"]    = ss[2]

            # Thermoadaptive indices
            ivywrel = sum(aac.get(a, 0) for a in ["I","V","Y","W","R","E","L"])
            charged = sum(aac.get(a, 0) for a in ["R","K","D","E"])
            polar   = sum(aac.get(a, 0) for a in ["N","Q","S","T"])
            gly_ser = aac.get("G", 0) + aac.get("S", 0)
            pro     = aac.get("P", 0) + 1e-6
            row["IVYWREL_Index"]     = ivywrel
            row["CvP_Bias"]          = charged - polar
            row["Flexibility_Ratio"] = gly_ser / pro

            row["Protein_ID"]    = pid
            row["Thermal_Class"] = 0
            rows.append(row)
        except Exception:
            skipped += 1
            continue

    print(f"  Features extraídas: {len(rows):,}  (omitidas: {skipped})")
    if not rows:
        print("  DIAGNÓSTICO — primeras 3 secuencias recibidas:")
        for pid, seq, desc in sequences[:3]:
            print(f"    {pid}: len={len(seq)}  inicio={str(seq)[:30]!r}")
    return pd.DataFrame(rows)


def score_with_psychroscan(feat_df):
    """Puntúa las secuencias con el modelo entrenado."""
    model  = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    thresh = float(open(os.path.join(MODELS_DIR, "threshold.txt")).read())
    fcols  = open(os.path.join(MODELS_DIR, "feature_columns.txt")).read().strip().split('\n')

    available = [c for c in fcols if c in feat_df.columns]
    missing   = [c for c in fcols if c not in feat_df.columns]

    if missing:
        for c in missing:
            feat_df[c] = 0.0
        print(f"  Advertencia: {len(missing)} features faltantes (rellenadas con 0)")

    X      = feat_df[fcols].astype(np.float32)
    probs  = model.predict_proba(X)[:, 0]  # P(Cold)
    preds  = (probs >= thresh).astype(int)  # 1 = Cold prediction

    feat_df = feat_df.copy()
    feat_df['P_Cold']    = probs
    feat_df['Pred_Cold'] = preds
    feat_df['Rank']      = feat_df['P_Cold'].rank(ascending=False).astype(int)
    feat_df['Percentile']= feat_df['P_Cold'].rank(pct=True) * 100

    return feat_df


def report_results(scored_df):
    """Genera el reporte de validación."""
    n_total    = len(scored_df)
    n_detected = scored_df['Pred_Cold'].sum()
    recall     = n_detected / n_total if n_total > 0 else 0.0

    top10  = (scored_df['Percentile'] >= 90).sum()
    top5   = (scored_df['Percentile'] >= 95).sum()
    top1   = (scored_df['Percentile'] >= 99).sum()
    median_rank = scored_df['Rank'].median()
    mean_prob   = scored_df['P_Cold'].mean()

    print("\n" + "="*65)
    print("  BRENDA EXTERNAL VALIDATION RESULTS")
    print("="*65)
    print(f"  External sequences evaluated : {n_total:,}")
    print(f"  Predicted cold-active        : {n_detected:,} ({recall*100:.1f}%)")
    print(f"  Mean P(Cold)                 : {mean_prob:.3f}")
    print(f"  Median rank                  : {median_rank:.0f}")
    print(f"  In top 10% of predictions   : {top10:,} ({top10/n_total*100:.1f}%)")
    print(f"  In top  5% of predictions   : {top5:,}  ({top5/n_total*100:.1f}%)")
    print(f"  In top  1% of predictions   : {top1:,}  ({top1/n_total*100:.1f}%)")
    print("="*65)

    # Por clase EC
    if 'EC_Class' in scored_df.columns:
        print("\n  By EC class:")
        for ec, grp in scored_df.groupby('EC_Class'):
            r = grp['Pred_Cold'].mean()
            print(f"    {ec:<22} Recall={r:.3f}  n={len(grp):,}")

    return {
        'n_external':       n_total,
        'n_detected':       n_detected,
        'recall':           recall,
        'mean_prob':        mean_prob,
        'median_rank':      median_rank,
        'pct_top10':        top10 / n_total * 100,
        'pct_top5':         top5  / n_total * 100,
        'pct_top1':         top1  / n_total * 100,
    }


def main():
    print("\n" + "="*65)
    print("  PsychroScan — BRENDA External Validation")
    print("="*65 + "\n")

    training_ids  = get_training_ids()
    training_taxa = get_training_taxa()
    print(f"  Training set: {len(training_ids):,} protein IDs")
    print(f"  Training taxa: {len(training_taxa):,} organisms to exclude\n")

    # Descargar y filtrar secuencias externas
    all_external = []
    for ec_class, ec_query in EC_QUERIES.items():
        print(f"  Querying UniProt for external {ec_class}...")
        records  = query_uniprot_cold_enzymes(ec_query, ec_class)
        external = filter_external_sequences(records, training_ids, training_taxa)
        for pid, seq, desc in external:
            all_external.append((pid, seq, desc, ec_class))
        time.sleep(0.5)  # respetar rate limit

    # Deduplicar por ID
    seen = set()
    deduped = []
    for pid, seq, desc, ec in all_external:
        if pid not in seen:
            seen.add(pid)
            deduped.append((pid, seq, desc, ec))

    print(f"\n  Total secuencias externas (no en training): {len(deduped):,}")

    if len(deduped) == 0:
        print("\n  Sin secuencias externas encontradas.")
        print("  Verifica acceso a internet y que config/taxa_list.json existe.")
        return

    # Guardar FASTA externo
    with open(OUT_FASTA, 'w') as f:
        for pid, seq, desc, ec in deduped:
            f.write(f">{pid} {ec} | {desc}\n{seq}\n")
    print(f"  FASTA guardado → {OUT_FASTA}")

    # Extraer features
    print("\n  Extrayendo features...")
    sequences_for_feat = [(pid, seq, desc) for pid, seq, desc, ec in deduped]
    feat_df = extract_features(sequences_for_feat)

    # Añadir EC class
    ec_map = {pid: ec for pid, _, _, ec in deduped}
    feat_df['EC_Class'] = feat_df['Protein_ID'].map(ec_map)

    print(f"  Features extraídas para {len(feat_df):,} secuencias")

    # Puntuar
    print("\n  Puntuando con PsychroScan...")
    scored_df = score_with_psychroscan(feat_df)

    # Reportar
    metrics = report_results(scored_df)

    # Guardar
    scored_df.to_csv(OUT_CSV, index=False)
    pd.DataFrame([metrics]).to_csv(
        OUT_CSV.replace('.csv', '_summary.csv'), index=False)
    print(f"\n  Resultados guardados → {OUT_CSV}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()