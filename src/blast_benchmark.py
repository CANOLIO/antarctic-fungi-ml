"""
PsychroScan — blast_benchmark.py (v2)
=======================================
Compara el recall de PsychroScan vs BLASTp sobre el hold-out test set.

FIX (v2):
  1. Antes reconstruía su propio split (comentario: "mismo split que
     05_train_model.py"), pero usaba dataset_features.csv mientras
     05_train_model.py (antes del fix) usaba dataset_features_nr90.csv —
     archivos distintos, splits no coincidentes. Ahora lee
     results/models/split_manifest.json (fuente única de verdad).
  2. La DB de referencia de BLAST se construía con TODAS las secuencias Cold,
     incluyendo las de organismos que ahora sabemos son parte del test set.
     Esto le daba a BLAST una ventaja injusta: podía encontrar hits casi
     idénticos (misma proteína/parálogo, mismo organismo) de las queries que
     se supone está "adivinando" sin haberlas visto. Ahora la DB se construye
     SOLO con secuencias de organismos de TRAIN.

Uso:
    python src/blast_benchmark.py
    (requiere BLAST+ instalado: makeblastdb, blastp)
"""

import os
import json
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import (roc_auc_score, precision_score,
                              recall_score, fbeta_score)
import joblib

# ─── RUTAS ────────────────────────────────────────────────────────────────────
DATA_FILE     = os.path.join("data", "processed", "dataset_features.csv")
RAW_DIR       = os.path.join("data", "raw", "industrial_enzymes")
MODELS_DIR    = os.path.join("results", "models")
MANIFEST_FILE = os.path.join(MODELS_DIR, "split_manifest.json")
BLAST_DB      = "/tmp/blast_psychro_db"
RESULTS_DIR   = os.path.join("results", "benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)

EVALUE_THRESHOLDS = [1e-3, 1e-5, 1e-10, 1e-20, 1e-50, 1e-100]


def load_manifest() -> dict:
    if not os.path.exists(MANIFEST_FILE):
        raise FileNotFoundError(
            f"No existe {MANIFEST_FILE}. Corre primero 05_train_model.py (v4)."
        )
    with open(MANIFEST_FILE) as f:
        return json.load(f)


def build_blast_db(train_protein_ids: set, force: bool = False):
    """
    Construye la DB SOLO con secuencias Cold de organismos de TRAIN.
    Antes se usaban TODAS las Cold_*.fasta, lo que incluía organismos del
    test set — BLAST podía "hacer trampa" encontrando su propia query.
    """
    db_path = BLAST_DB + ".psq"
    if os.path.exists(db_path) and not force:
        print("  BLAST DB ya existe — omitiendo construcción (usa force=True para regenerar).")
        return

    print("  Construyendo BLAST DB SOLO con secuencias Cold de organismos de TRAIN...")
    cold_fastas = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR)
                   if f.startswith("Cold_") and f.endswith(".fasta")]

    merged = "/tmp/psychrophile_db.fasta"
    n_written, n_skipped = 0, 0
    with open(merged, 'w') as out:
        for fp in cold_fastas:
            for rec in SeqIO.parse(fp, 'fasta'):
                if rec.id in train_protein_ids:
                    out.write(f">{rec.description}\n{str(rec.seq)}\n")
                    n_written += 1
                else:
                    n_skipped += 1

    print(f"    Secuencias incluidas (train): {n_written:,}")
    print(f"    Secuencias excluidas (test o no-Cold): {n_skipped:,}")

    subprocess.run([
        "makeblastdb", "-in", merged, "-dbtype", "prot",
        "-out", BLAST_DB, "-title", "PsychroScan_Cold_Reference_TRAIN_ONLY",
    ], check=True)
    print(f"  DB construida: {BLAST_DB}")


def get_test_df(manifest: dict) -> pd.DataFrame:
    """Carga el test set EXACTO usado para entrenar (desde el manifiesto)."""
    df = pd.read_csv(DATA_FILE)
    test_ids = set(manifest['test_protein_ids'])
    test_df = df[df['Protein_ID'].isin(test_ids)][
        ['Protein_ID', 'Thermal_Class', 'EC_Class', 'Organism_Source']
    ].reset_index(drop=True)
    print(f"  Test set (organismos held-out, {len(manifest['test_organisms'])} organismos): "
          f"{len(test_df):,} proteínas "
          f"(Cold={(test_df['Thermal_Class']==0).sum():,}, "
          f"Warm={(test_df['Thermal_Class']==1).sum():,})")
    return test_df


def extract_test_fasta(test_df):
    test_ids = set(test_df['Protein_ID'].tolist())
    found    = {}
    print(f"  Extrayendo {len(test_ids):,} secuencias del test set...")
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith('.fasta'):
            continue
        for rec in SeqIO.parse(os.path.join(RAW_DIR, fname), 'fasta'):
            if rec.id in test_ids and rec.id not in found:
                found[rec.id] = str(rec.seq)
        if len(found) == len(test_ids):
            break
    print(f"  Encontradas: {len(found):,} / {len(test_ids):,}")
    fasta_path = "/tmp/test_set_queries.fasta"
    with open(fasta_path, 'w') as f:
        for pid, seq in found.items():
            f.write(f">{pid}\n{seq}\n")
    return fasta_path, found


def run_blastp(query_fasta, n_threads=4):
    out_path = "/tmp/blast_results.tsv"
    print(f"  Corriendo BLASTp (esto puede tardar 5-15 min)...")
    subprocess.run([
        "blastp", "-query", query_fasta, "-db", BLAST_DB, "-out", out_path,
        "-outfmt", "6 qseqid sseqid evalue bitscore pident",
        "-evalue", "10", "-num_threads", str(n_threads), "-max_target_seqs", "1",
    ], check=True)
    cols = ['query_id', 'subject_id', 'evalue', 'bitscore', 'pident']
    if os.path.getsize(out_path) == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(out_path, sep='\t', names=cols)
    df = df.sort_values('evalue').drop_duplicates('query_id')
    print(f"  BLAST: {len(df):,} queries con al menos un hit")
    return df


def evaluate_blast(test_df, blast_df, evalue_thresh):
    hit_ids = set(blast_df[blast_df['evalue'] <= evalue_thresh]['query_id'].tolist())
    merged  = test_df.copy()
    merged['blast_pred'] = merged['Protein_ID'].apply(lambda pid: 0 if pid in hit_ids else 1)
    true_cold = 1 - merged['Thermal_Class'].values
    pred_cold = 1 - merged['blast_pred'].values
    blast_scores = merged['Protein_ID'].map(
        blast_df.set_index('query_id')['bitscore'].to_dict()).fillna(0).values
    try:
        auc = roc_auc_score(true_cold, blast_scores)
    except Exception:
        auc = float('nan')
    return {
        'evalue_threshold': evalue_thresh,
        'recall_cold':    recall_score(true_cold, pred_cold, zero_division=0),
        'precision_cold': precision_score(true_cold, pred_cold, zero_division=0),
        'f2_cold':        fbeta_score(true_cold, pred_cold, beta=2, zero_division=0),
        'auc_roc':        auc,
        'n_hits':         len(hit_ids),
    }


def get_psychroscan_metrics(test_df):
    """Métricas de PsychroScan sobre el MISMO test set (manifiesto), sin re-derivar split."""
    model  = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    thresh = float(open(os.path.join(MODELS_DIR, "threshold.txt")).read())
    fcols  = open(os.path.join(MODELS_DIR, "feature_columns.txt")).read().strip().split('\n')

    df_all = pd.read_csv(DATA_FILE)
    df_te  = df_all[df_all['Protein_ID'].isin(set(test_df['Protein_ID']))]
    X_te   = df_te[fcols].astype(np.float32)
    y_te   = df_te['Thermal_Class'].values

    probs_cold = model.predict_proba(X_te)[:, 0]
    y_pred     = 1 - (probs_cold >= thresh).astype(int)
    true_cold  = 1 - y_te
    pred_cold  = 1 - y_pred
    return {
        'method':         'PsychroScan (LightGBM)',
        'recall_cold':    recall_score(true_cold, pred_cold),
        'precision_cold': precision_score(true_cold, pred_cold),
        'f2_cold':        fbeta_score(true_cold, pred_cold, beta=2),
        'auc_roc':        roc_auc_score(true_cold, probs_cold),
    }


def main():
    print("\n" + "="*65)
    print("  PsychroScan vs BLASTp — Benchmark Comparison (v2)")
    print("  DB de BLAST ahora excluye organismos del test set")
    print("="*65 + "\n")

    manifest = load_manifest()
    test_df  = get_test_df(manifest)

    train_protein_ids = set(manifest['train_protein_ids'])
    build_blast_db(train_protein_ids)

    query_fasta, _ = extract_test_fasta(test_df)
    blast_df = run_blastp(query_fasta)

    print("\n  Evaluando BLAST a distintos E-value thresholds...")
    blast_results = []
    for thresh in EVALUE_THRESHOLDS:
        metrics = evaluate_blast(test_df, blast_df, thresh)
        blast_results.append(metrics)
        print(f"  E-value <= {thresh:.0e}: "
              f"Recall={metrics['recall_cold']:.3f}  "
              f"Precision={metrics['precision_cold']:.3f}  "
              f"AUC={metrics['auc_roc']:.4f}  "
              f"Hits={metrics['n_hits']:,}")

    print("\n  Calculando métricas PsychroScan...")
    ps_metrics = get_psychroscan_metrics(test_df)

    best_blast = max(blast_results, key=lambda x: x['recall_cold'])
    print("\n" + "="*65)
    print("  COMPARISON TABLE (held-out por organismo)")
    print("="*65)
    print(f"  {'Method':<30} {'AUC':>7}  {'Recall':>7}  {'Prec.':>7}  {'F2':>7}")
    print(f"  {'-'*60}")
    print(f"  {'PsychroScan (LightGBM)':<30} "
          f"{ps_metrics['auc_roc']:>7.4f}  {ps_metrics['recall_cold']:>7.3f}  "
          f"{ps_metrics['precision_cold']:>7.3f}  {ps_metrics['f2_cold']:>7.3f}")
    print(f"  {'BLASTp (best threshold)':<30} "
          f"{best_blast['auc_roc']:>7.4f}  {best_blast['recall_cold']:>7.3f}  "
          f"{best_blast['precision_cold']:>7.3f}  {best_blast['f2_cold']:>7.3f}")

    rows = [{'method': 'PsychroScan', **ps_metrics}]
    for r in blast_results:
        rows.append({'method': f"BLASTp E<={r['evalue_threshold']:.0e}", **r})
    out = os.path.join(RESULTS_DIR, 'blast_benchmark.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Tabla guardada → {out}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
