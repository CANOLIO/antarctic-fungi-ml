import os
import csv
import sys
import argparse
from itertools import product
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ─── RUTAS ────────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw", "industrial_enzymes")
PROCESSED_DIR = os.path.join("data", "processed")
OUT_FILE      = os.path.join(PROCESSED_DIR, "dataset_features.csv")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─── CONSTANTES ───────────────────────────────────────────────────────────────
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES  = [''.join(p) for p in product(AMINO_ACIDS, repeat=2)]
VALID_AA    = set(AMINO_ACIDS)
BATCH_SIZE  = 2000

# Residuos para features termoadaptativos
IVYWREL_SET  = set("IVYWREL")          # Zeldovich 2007
CHARGED_SET  = set("RKDE")            # Cargados a pH fisiológico
POLAR_SET    = set("NQST")            # Polares sin carga

# ─── ENCABEZADO CSV ───────────────────────────────────────────────────────────
HEADER = (
    ['Protein_ID', 'Organism_Source', 'EC_Class', 'Thermal_Class',
     'Length', 'Molecular_Weight', 'GRAVY', 'Instability_Index',
     'Aromaticity', 'Helix_Fraction', 'Turn_Fraction', 'Sheet_Fraction',
     # ── Nuevos features termoadaptativos (v3) ────────────────────────────────
     'IVYWREL_Index',      # Predictor universal T°; frío=bajo, cálido=alto
     'CvP_Bias',           # Charged vs Polar; frío=negativo
     'Flexibility_Ratio',  # (Gly+Ser)/Pro; frío=alto
     ]
    + [f'AAC_{aa}' for aa in AMINO_ACIDS]
    + [f'DPC_{di}' for di in DIPEPTIDES]
)


def compute_thermal_features(seq: str) -> dict:
    """
    Calcula los 3 índices termoadaptativos descritos en el docstring del módulo.
    Entrada: secuencia ya validada (sólo AA estándar, len >= 20).
    """
    slen = len(seq)

    # IVYWREL Index
    ivywrel_count = sum(1 for aa in seq if aa in IVYWREL_SET)
    ivywrel_index = ivywrel_count / slen

    # CvP Bias
    charged_count = sum(1 for aa in seq if aa in CHARGED_SET)
    polar_count   = sum(1 for aa in seq if aa in POLAR_SET)
    cvp_bias      = (charged_count - polar_count) / slen

    # Flexibility Ratio
    gly_count     = seq.count('G')
    ser_count     = seq.count('S')
    pro_count     = seq.count('P')
    flex_ratio    = (gly_count + ser_count) / (pro_count + 0.001)

    return {
        'IVYWREL_Index':     ivywrel_index,
        'CvP_Bias':          cvp_bias,
        'Flexibility_Ratio': flex_ratio,
    }


def extract_features(seq: str, record_id: str, organism: str,
                     ec_class: str, thermal_class: int) -> dict | None:
    """Extrae 431 features (428 originales + 3 termoadaptativos nuevos)."""
    seq = seq.upper()
    if not set(seq).issubset(VALID_AA) or len(seq) < 20:
        return None

    try:
        analysis = ProteinAnalysis(seq)
        ss       = analysis.secondary_structure_fraction()
        aac      = analysis.amino_acids_percent
        seq_len  = len(seq)

        feat = {
            'Protein_ID':       record_id,
            'Organism_Source':  organism,
            'EC_Class':         ec_class,       # ← bug corregido vs v2
            'Thermal_Class':    thermal_class,
            'Length':           seq_len,
            'Molecular_Weight': analysis.molecular_weight(),
            'GRAVY':            analysis.gravy(),
            'Instability_Index':analysis.instability_index(),
            'Aromaticity':      analysis.aromaticity(),
            'Helix_Fraction':   ss[0],
            'Turn_Fraction':    ss[1],
            'Sheet_Fraction':   ss[2],
        }

        # Nuevos features termoadaptativos
        feat.update(compute_thermal_features(seq))

        # AAC y DPC originales
        for aa in AMINO_ACIDS:
            feat[f'AAC_{aa}'] = aac[aa]
        for di in DIPEPTIDES:
            feat[f'DPC_{di}'] = seq.count(di) / (seq_len - 1)

        return feat

    except Exception:
        return None


def main(force: bool = False):
    # ── Idempotencia ──────────────────────────────────────────────────────────
    if os.path.exists(OUT_FILE) and not force:
        size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
        print(f"\n⚠️  Ya existe {OUT_FILE} ({size_mb:.1f} MB).")
        ans = input("   ¿Re-generar desde cero? [s/N]: ").strip().lower()
        if ans != 's':
            print("   Cancelado. Usa --force para omitir esta pregunta.")
            sys.exit(0)
        os.remove(OUT_FILE)
        print("   Archivo anterior borrado. Re-extrayendo...\n")

    fastas = sorted(f for f in os.listdir(RAW_DIR) if f.endswith('.fasta'))
    if not fastas:
        print(f"❌ No se encontraron .fasta en {RAW_DIR}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  PsychroScan — Extracción de Features v3")
    print("  Nuevos: IVYWREL_Index, CvP_Bias, Flexibility_Ratio")
    print("=" * 70)
    print(f"  Input  : {RAW_DIR}/")
    print(f"  Output : {OUT_FILE}")
    print(f"  Archivos FASTA: {len(fastas)}")
    for f in fastas:
        size_kb = os.path.getsize(os.path.join(RAW_DIR, f)) / 1024
        print(f"    · {f:<42} {size_kb:>6.0f} KB")
    print("=" * 70 + "\n")

    total_valid   = 0
    total_skipped = 0
    class_counts  = {0: 0, 1: 0}
    ec_counts     = {}

    # Estadísticas de los nuevos features (para validar que tienen sentido)
    thermal_stats = {0: {'ivywrel': [], 'cvp': [], 'flex': []},
                     1: {'ivywrel': [], 'cvp': [], 'flex': []}}

    with open(OUT_FILE, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HEADER)
        writer.writeheader()

        for filename in fastas:
            base  = filename.replace('.fasta', '')
            parts = base.split('_', 1)

            if len(parts) < 2 or parts[0] not in ('Cold', 'Warm'):
                print(f"⚠️  Nombre inesperado: {filename} — omitiendo.")
                continue

            thermal_class = 0 if parts[0] == 'Cold' else 1
            ec_class_name = parts[1]
            filepath      = os.path.join(RAW_DIR, filename)

            print(f"Procesando [{parts[0].upper()} / {ec_class_name}] → {filename}...")

            batch      = []
            file_valid = 0
            file_skip  = 0

            for record in SeqIO.parse(filepath, 'fasta'):
                feat = extract_features(
                    str(record.seq), record.id,
                    ec_class_name, ec_class_name, thermal_class  # bug v2 corregido
                )
                if feat is None:
                    file_skip += 1
                    continue

                batch.append(feat)
                file_valid += 1

                # Acumular para estadísticas (solo una muestra, no toda la RAM)
                if file_valid <= 500:
                    tc = thermal_class
                    thermal_stats[tc]['ivywrel'].append(feat['IVYWREL_Index'])
                    thermal_stats[tc]['cvp'].append(feat['CvP_Bias'])
                    thermal_stats[tc]['flex'].append(feat['Flexibility_Ratio'])

                if len(batch) >= BATCH_SIZE:
                    writer.writerows(batch)
                    batch = []

            if batch:
                writer.writerows(batch)

            total_valid   += file_valid
            total_skipped += file_skip
            class_counts[thermal_class] = class_counts.get(thermal_class, 0) + file_valid
            ec_counts[ec_class_name]    = ec_counts.get(ec_class_name, 0) + file_valid

            print(f"  ✅ {file_valid:,} válidas  |  {file_skip} omitidas\n")

    # ── Resumen final ─────────────────────────────────────────────────────────
    import statistics
    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    ratio   = class_counts[1] / max(class_counts[0], 1)

    print("=" * 70)
    print("  RESUMEN FINAL")
    print("=" * 70)
    print(f"  Total válidas  : {total_valid:,}")
    print(f"  Total omitidas : {total_skipped:,}")
    print(f"  CSV            : {size_mb:.2f} MB  →  {OUT_FILE}")
    print()
    print(f"  BALANCE:")
    print(f"    ❄️  Cold (0) : {class_counts[0]:,} proteínas")
    print(f"    🌱  Warm (1) : {class_counts[1]:,} proteínas")
    print(f"    Ratio        : {ratio:.1f}x")

    # Validación biológica de los nuevos features
    print()
    print("  VALIDACIÓN DE FEATURES TERMOADAPTATIVOS")
    print("  (Si el modelo es correcto: Cold < Warm para IVYWREL y CvP)")
    print("  " + "-" * 50)

    for label, tc in [("❄️  Cold", 0), ("🌱  Warm", 1)]:
        s = thermal_stats[tc]
        if s['ivywrel']:
            mean_iv  = statistics.mean(s['ivywrel'])
            mean_cvp = statistics.mean(s['cvp'])
            mean_fl  = statistics.mean(s['flex'])
            print(f"  {label}:")
            print(f"    IVYWREL_Index    = {mean_iv:.4f}  {'← esperado menor' if tc==0 else '← esperado mayor'}")
            print(f"    CvP_Bias         = {mean_cvp:.4f}  {'← esperado menor/neg.' if tc==0 else '← esperado mayor/pos.'}")
            print(f"    Flexibility_Ratio= {mean_fl:.4f}  {'← esperado mayor' if tc==0 else '← esperado menor'}")

    print()
    print("  DISTRIBUCIÓN POR EC:")
    for ec, n in sorted(ec_counts.items()):
        print(f"    · {ec:<25} {n:>5,} proteínas")

    print()
    print("  ✅ Siguiente paso → 05_train_model.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PsychroScan Feature Extraction v3")
    parser.add_argument('--force', action='store_true',
                        help='Sobreescribir sin preguntar')
    args = parser.parse_args()
    main(force=args.force)