"""
PsychroScan — blast_benchmark.py
==================================
Compara el recall de PsychroScan vs BLASTp sobre el hold-out test set.

Flujo:
  1. Reconstruye el test set (mismo split que 05_train_model.py)
  2. Extrae secuencias FASTA del test set desde los FASTAs crudos
  3. Corre BLASTp contra base de datos de secuencias psicrófilas de entrenamiento
  4. Clasifica: hit con E-value <= umbral → BLAST predice "Cold"
  5. Compara recall, precision, AUC vs PsychroScan

Uso:
    # Primero construir la DB (una sola vez):
    cat data/raw/industrial_enzymes/Cold_*.fasta > /tmp/psychrophile_db.fasta
    makeblastdb -in /tmp/psychrophile_db.fasta -dbtype prot -out /tmp/blast_psychro_db

    # Luego correr este script:
    python src/blast_benchmark.py
"""

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score,
                              recall_score, fbeta_score)
import joblib

# ─── RUTAS ────────────────────────────────────────────────────────────────────
DATA_FILE    = os.path.join("data", "processed", "dataset_features.csv")
RAW_DIR      = os.path.join("data", "raw", "industrial_enzymes")
MODELS_DIR   = os.path.join("results", "models")
BLAST_DB     = "/tmp/blast_psychro_db"
RESULTS_DIR  = os.path.join("results", "benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)

# E-value thresholds to sweep for BLAST ROC
EVALUE_THRESHOLDS = [1e-3, 1e-5, 1e-10, 1e-20, 1e-50, 1e-100]
RANDOM_STATE      = 42


def build_blast_db():
    """Construye la DB si no existe."""
    db_path = BLAST_DB + ".psq"
    if os.path.exists(db_path):
        print("  BLAST DB ya existe — omitiendo construcción.")
        return
    print("  Construyendo BLAST DB de secuencias psicrófilas...")
    cold_fastas = [os.path.join(RAW_DIR, f)
                   for f in os.listdir(RAW_DIR)
                   if f.startswith("Cold_") and f.endswith(".fasta")]
    merged = "/tmp/psychrophile_db.fasta"
    with open(merged, 'w') as out:
        for fp in cold_fastas:
            with open(fp) as f:
                out.write(f.read())
    subprocess.run([
        "makeblastdb", "-in", merged,
        "-dbtype", "prot",
        "-out", BLAST_DB,
        "-title", "PsychroScan_Cold_Reference"
    ], check=True)
    print(f"  DB construida: {BLAST_DB}")


def get_test_ids():
    """Reproduce el mismo test split que 05_train_model.py."""
    df = pd.read_csv(DATA_FILE)
    df['strat_key'] = df['EC_Class'] + "_" + df['Thermal_Class'].astype(str)
    counts = df['strat_key'].value_counts()
    rare   = counts[counts < 5].index.tolist()
    df     = df[~df['strat_key'].isin(rare)]
    feat_cols = [c for c in df.columns
                 if c not in ['Protein_ID','Organism_Source','EC_Class',
                               'Thermal_Class','strat_key']]
    X = df[feat_cols].astype(np.float32)
    y = df['Thermal_Class'].values
    _, X_te, _, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index,
        test_size=0.20,
        stratify=df['strat_key'],
        random_state=RANDOM_STATE,
    )
    test_df = df.loc[idx_te, ['Protein_ID', 'Thermal_Class', 'EC_Class']].reset_index(drop=True)
    print(f"  Test set: {len(test_df):,} proteínas "
          f"(Cold={( test_df['Thermal_Class']==0).sum():,}, "
          f"Warm={(test_df['Thermal_Class']==1).sum():,})")
    return test_df


def extract_test_fasta(test_df):
    """Extrae secuencias FASTA del test set desde los FASTAs crudos."""
    test_ids = set(test_df['Protein_ID'].tolist())
    found    = {}
    print(f"  Extrayendo {len(test_ids):,} secuencias del test set...")
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith('.fasta'):
            continue
        fpath = os.path.join(RAW_DIR, fname)
        for rec in SeqIO.parse(fpath, 'fasta'):
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
    """Corre BLASTp y devuelve DataFrame con hits."""
    out_path = "/tmp/blast_results.tsv"
    print(f"  Corriendo BLASTp (esto puede tardar 5-15 min)...")
    subprocess.run([
        "blastp",
        "-query",    query_fasta,
        "-db",       BLAST_DB,
        "-out",      out_path,
        "-outfmt",   "6 qseqid sseqid evalue bitscore pident",
        "-evalue",   "10",          # umbral permisivo — filtraremos después
        "-num_threads", str(n_threads),
        "-max_target_seqs", "1",    # solo el mejor hit por query
    ], check=True)
    cols = ['query_id', 'subject_id', 'evalue', 'bitscore', 'pident']
    if os.path.getsize(out_path) == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(out_path, sep='\t', names=cols)
    # Un query puede tener múltiples líneas si max_target_seqs > 1
    # Tomar solo el mejor hit (menor evalue) por query
    df = df.sort_values('evalue').drop_duplicates('query_id')
    print(f"  BLAST: {len(df):,} queries con al menos un hit")
    return df


def evaluate_blast(test_df, blast_df, evalue_thresh):
    """
    Clasifica: si query tiene hit con evalue <= thresh → pred Cold (0).
    Sin hit → pred Warm (1) (BLAST no encontró homólogo psicrófilo).
    """
    hit_ids = set(blast_df[blast_df['evalue'] <= evalue_thresh]['query_id'].tolist())
    merged  = test_df.copy()
    merged['blast_pred'] = merged['Protein_ID'].apply(
        lambda pid: 0 if pid in hit_ids else 1)
    true_cold  = 1 - merged['Thermal_Class'].values
    pred_cold  = 1 - merged['blast_pred'].values
    # Usar bitscore como proxy de score para AUC (mayor bitscore = más "cold")
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
    """Obtiene métricas de PsychroScan sobre el mismo test set."""
    model  = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    thresh = float(open(os.path.join(MODELS_DIR, "threshold.txt")).read())
    fcols  = open(os.path.join(MODELS_DIR, "feature_columns.txt")).read().strip().split('\n')
    df_all = pd.read_csv(DATA_FILE)
    df_all['strat_key'] = df_all['EC_Class'] + "_" + df_all['Thermal_Class'].astype(str)
    counts = df_all['strat_key'].value_counts()
    rare   = counts[counts < 5].index.tolist()
    df_all = df_all[~df_all['strat_key'].isin(rare)]
    X_all  = df_all[[c for c in fcols if c in df_all.columns]].astype(np.float32)
    y_all  = df_all['Thermal_Class'].values
    _, X_te, _, y_te = train_test_split(
        X_all, y_all, test_size=0.20,
        stratify=df_all['strat_key'], random_state=RANDOM_STATE)
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
    print("  PsychroScan vs BLASTp — Benchmark Comparison")
    print("="*65 + "\n")

    # 1. Construir DB
    build_blast_db()

    # 2. Test set
    test_df = get_test_ids()

    # 3. Extraer FASTAs
    query_fasta, _ = extract_test_fasta(test_df)

    # 4. Correr BLAST
    blast_df = run_blastp(query_fasta)

    # 5. Evaluar BLAST a distintos umbrales
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

    # 6. Métricas PsychroScan
    print("\n  Calculando métricas PsychroScan...")
    ps_metrics = get_psychroscan_metrics(test_df)

    # 7. Tabla comparativa
    best_blast = max(blast_results, key=lambda x: x['recall_cold'])
    print("\n" + "="*65)
    print("  COMPARISON TABLE")
    print("="*65)
    print(f"  {'Method':<30} {'AUC':>7}  {'Recall':>7}  {'Prec.':>7}  {'F2':>7}")
    print(f"  {'-'*60}")
    print(f"  {'PsychroScan (LightGBM)':<30} "
          f"{ps_metrics['auc_roc']:>7.4f}  "
          f"{ps_metrics['recall_cold']:>7.3f}  "
          f"{ps_metrics['precision_cold']:>7.3f}  "
          f"{ps_metrics['f2_cold']:>7.3f}")
    print(f"  {'BLASTp (best threshold)':<30} "
          f"{best_blast['auc_roc']:>7.4f}  "
          f"{best_blast['recall_cold']:>7.3f}  "
          f"{best_blast['precision_cold']:>7.3f}  "
          f"{best_blast['f2_cold']:>7.3f}")

    # 8. Guardar CSV
    rows = [{'method': 'PsychroScan', **ps_metrics}]
    for r in blast_results:
        rows.append({'method': f"BLASTp E<={r['evalue_threshold']:.0e}", **r})
    out = os.path.join(RESULTS_DIR, 'blast_benchmark.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Tabla guardada → {out}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()