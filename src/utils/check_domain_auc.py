"""
PsychroScan — check_domain_auc.py
==================================
Chequeo de robustez taxonómica sobre el test set held-out (n=621, 50 organismos).
Estratifica el AUC por dominio biológico (Bacteria vs. Fungi) para verificar
si el modelo reconoce adaptación al frío intra-dominio o si sufre sesgos filogenéticos.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, fbeta_score, recall_score, precision_score, accuracy_score

DATA_FILE     = os.path.join("data", "processed", "dataset_features.csv")
MODELS_DIR    = os.path.join("results", "models")
MANIFEST_FILE = os.path.join(MODELS_DIR, "split_manifest.json")
FEATURE_FILE  = os.path.join(MODELS_DIR, "feature_columns.txt")
MODEL_FILE    = os.path.join(MODELS_DIR, "optuna_f2_model.pkl")

FUNGI_GENERA = {
    'saccharomyces', 'schizosaccharomyces', 'candida', 'aspergillus', 'neurospora',
    'trichoderma', 'botrytis', 'ustilago', 'magnaporthe', 'yarrowia', 'rhodotorula',
    'leucosporidium', 'glaciozyma', 'mrakia', 'cryomyces', 'pseudogymnoascus',
    'thelebolus', 'phenoliferia', 'goffeauzyma', 'guehomyces', 'tausonia',
    'naganishia', 'geomyces', 'cladosporium', 'penicillium', 'geotrichum',
    'pyricularia', 'emericella', 'mycosarcoma', 'dioszegia', 'sungouiella'
}


def get_domain(org_str: str) -> str:
    org_lower = str(org_str).lower()
    if any(g in org_lower for g in FUNGI_GENERA):
        return 'Fungi'
    if 'methano' in org_lower:
        return 'Archaea'
    return 'Bacteria'


def main():
    print("\n" + "=" * 70)
    print("  PsychroScan — Robustez Taxonómica por Dominio (Held-Out Test Set)")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    with open(FEATURE_FILE) as f:
        feat_cols = [line.strip() for line in f if line.strip()]

    model = joblib.load(MODEL_FILE)
    tau = manifest['threshold']

    test_df = df[df['Protein_ID'].isin(manifest['test_protein_ids'])].copy()
    X_test = test_df[feat_cols].astype(np.float32)

    test_df['prob_cold'] = model.predict_proba(X_test)[:, 0]
    test_df['pred_cold'] = (test_df['prob_cold'] >= tau).astype(int)
    test_df['y_cold'] = (test_df['Thermal_Class'] == 0).astype(int)
    test_df['Domain'] = test_df['Organism_Source'].apply(get_domain)

    print(f"\n  Total secuencias en test set held-out : {len(test_df):,}")
    print(f"  Organismos únicos en test set        : {test_df['Organism_Source'].nunique()}")
    print(f"  Umbral calibrado en Train (tau)      : {tau:.4f}\n")

    print("  Distribución de clases por dominio:")
    dist = test_df.groupby(['Domain', 'Thermal_Class']).size().unstack(fill_value=0)
    dist.columns = ['Cold (Class 0)', 'Warm (Class 1)']
    print(dist.to_string())

    print("\n" + "-" * 70)
    print("  MÉTRICAS POR DOMINIO TAXONÓMICO")
    print("-" * 70)

    for dom in ['Bacteria', 'Fungi']:
        sub = test_df[test_df['Domain'] == dom]
        if len(sub) == 0 or sub['y_cold'].nunique() < 2:
            continue

        auc = roc_auc_score(sub['y_cold'], sub['prob_cold'])
        f2 = fbeta_score(sub['y_cold'], sub['pred_cold'], beta=2)
        rec = recall_score(sub['y_cold'], sub['pred_cold'])
        prec = precision_score(sub['y_cold'], sub['pred_cold'])
        acc = accuracy_score(sub['y_cold'], sub['pred_cold'])
        n_cold = (sub['y_cold'] == 1).sum()
        n_warm = (sub['y_cold'] == 0).sum()

        print(f"\n  Dominio: {dom.upper()} (n={len(sub):,} | Cold={n_cold}, Warm={n_warm})")
        print(f"    ROC-AUC   : {auc:.4f}")
        print(f"    F2-Score  : {f2:.4f}")
        print(f"    Recall    : {rec:.4f} ({rec*100:.1f}%)")
        print(f"    Precision : {prec:.4f} ({prec*100:.1f}%)")
        print(f"    Accuracy  : {acc:.4f} ({acc*100:.1f}%)")

    auc_overall = roc_auc_score(test_df['y_cold'], test_df['prob_cold'])
    print("\n" + "=" * 70)
    print(f"  ROC-AUC Global en Test Held-Out (n=621): {auc_overall:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
