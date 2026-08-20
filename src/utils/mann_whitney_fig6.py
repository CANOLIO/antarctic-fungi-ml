
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

DATA_FILE = os.path.join("data", "processed", "dataset_features.csv")
OUT_DIR   = os.path.join("results", "benchmark")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

if 'Organism_Source' not in df.columns:
    raise ValueError(
        "No existe 'Organism_Source'. Este script requiere el CSV generado "
        "por la versión corregida de 03_feature_extraction.py."
    )

AA_LABELS = [('AAC_G', 'Glycine'), ('AAC_S', 'Serine'), ('AAC_P', 'Proline')]


def sig_stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def run_mannwhitney(cold_vals, warm_vals):
    stat, p = mannwhitneyu(cold_vals, warm_vals, alternative='two-sided')
    n1, n2 = len(cold_vals), len(warm_vals)
    r = 1 - (2 * stat) / (n1 * n2)
    return stat, p, r


print("=" * 68)
print("  Mann-Whitney U — Amino Acid Composition (Fig. 6)")
print("  Reportando nivel PROTEÍNA (pseudo-replicado) y nivel ORGANISMO (válido)")
print("=" * 68)

results = []

# ── Nivel proteína (legacy, pseudo-replicado) ─────────────────────────────
cold_p = df[df['Thermal_Class'] == 0]
warm_p = df[df['Thermal_Class'] == 1]
print(f"\n  [Nivel proteína — pseudo-replicado] n Cold={len(cold_p):,}  n Warm={len(warm_p):,}")

# ── Nivel organismo (válido): una fila = media de un organismo ────────────
org_means = (df.groupby(['Organism_Source', 'Thermal_Class'])[['AAC_G', 'AAC_S', 'AAC_P']]
             .mean().reset_index())
cold_o = org_means[org_means['Thermal_Class'] == 0]
warm_o = org_means[org_means['Thermal_Class'] == 1]
print(f"  [Nivel organismo — válido]         n Cold={len(cold_o)}      n Warm={len(warm_o)}\n")

for aa, label in AA_LABELS:
    # Proteína
    stat_p, p_p, r_p = run_mannwhitney(cold_p[aa].values * 100, warm_p[aa].values * 100)
    # Organismo
    stat_o, p_o, r_o = run_mannwhitney(cold_o[aa].values * 100, warm_o[aa].values * 100)

    agreement = "✅ concuerdan" if (np.sign(r_p) == np.sign(r_o)) else "🚨 signo distinto — revisar"

    print(f"  {label} ({aa}):")
    print(f"    Proteína  (n pseudo-replicado) : p={p_p:.2e} {sig_stars(p_p):<4} r={r_p:+.3f}")
    print(f"    Organismo (n real = {len(cold_o)+len(warm_o)})        : p={p_o:.2e} {sig_stars(p_o):<4} r={r_o:+.3f}")
    print(f"    → {agreement}\n")

    results.append({
        'Residue': label,
        'Protein_level_p': f"{p_p:.2e}", 'Protein_level_r': round(r_p, 4),
        'Protein_level_sig': sig_stars(p_p),
        'Organism_level_p': f"{p_o:.2e}", 'Organism_level_r': round(r_o, 4),
        'Organism_level_sig': sig_stars(p_o),
        'Organism_level_n_cold': len(cold_o), 'Organism_level_n_warm': len(warm_o),
        'Same_direction': bool(np.sign(r_p) == np.sign(r_o)),
    })

out_path = os.path.join(OUT_DIR, 'mannwhitney_fig6.csv')
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"  Tabla guardada → {out_path}")
print("  Para el paper: cita el nivel ORGANISMO como prueba primaria; el nivel")
print("  proteína puede mencionarse como corroborativo, nunca como el N real.")
print("=" * 68)
