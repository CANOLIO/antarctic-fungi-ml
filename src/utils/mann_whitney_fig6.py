"""
Mann-Whitney U tests para Fig. 6 — Gly, Ser, Pro
Corre desde la raíz del proyecto:
    python src/mann_whitney_fig6.py
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

df = pd.read_csv('data/processed/dataset_features.csv')

cold = df[df['Thermal_Class'] == 0]
warm = df[df['Thermal_Class'] == 1]

print("=" * 60)
print("  Mann-Whitney U — Amino Acid Composition (Fig. 6)")
print("=" * 60)
print(f"  n Cold = {len(cold):,}   n Warm = {len(warm):,}\n")

results = []
for aa, label in [('AAC_G', 'Glycine'), ('AAC_S', 'Serine'), ('AAC_P', 'Proline')]:
    cold_vals = cold[aa].values * 100
    warm_vals = warm[aa].values * 100

    stat, p = mannwhitneyu(cold_vals, warm_vals, alternative='two-sided')

    # Effect size: rank-biserial correlation
    n1, n2  = len(cold_vals), len(warm_vals)
    r       = 1 - (2 * stat) / (n1 * n2)

    # Resumen
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {label} ({aa}):")
    print(f"    Cold median : {np.median(cold_vals):.2f}%")
    print(f"    Warm median : {np.median(warm_vals):.2f}%")
    print(f"    U statistic : {stat:.0f}")
    print(f"    p-value     : {p:.2e}  {sig}")
    print(f"    Effect size : r = {r:.4f}")
    print()
    results.append({
        'Residue': label,
        'Cold_median_%': round(np.median(cold_vals), 3),
        'Warm_median_%': round(np.median(warm_vals), 3),
        'U_statistic':   round(stat),
        'p_value':       f"{p:.2e}",
        'r_effect_size': round(r, 4),
        'Significance':  sig,
    })

import os
os.makedirs('results/benchmark', exist_ok=True)
pd.DataFrame(results).to_csv('results/benchmark/mannwhitney_fig6.csv', index=False)
print("  Tabla guardada → results/benchmark/mannwhitney_fig6.csv")
print("=" * 60)