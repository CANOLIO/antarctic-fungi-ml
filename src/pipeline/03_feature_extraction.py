import os
import csv
import sys
import argparse
from itertools import product
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.organism_resolver import load_taxa_config, resolve_organism

# ─── RUTAS ────────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw", "industrial_enzymes")
PROCESSED_DIR = os.path.join("data", "processed")
OUT_FILE      = os.path.join(PROCESSED_DIR, "dataset_features.csv")
TAXA_CONFIG   = os.path.join("config", "taxa_list.json")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─── CONSTANTES ───────────────────────────────────────────────────────────────
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES  = [''.join(p) for p in product(AMINO_ACIDS, repeat=2)]
VALID_AA    = set(AMINO_ACIDS)
BATCH_SIZE  = 2000

IVYWREL_SET  = set("IVYWREL")
CHARGED_SET  = set("RKDE")
POLAR_SET    = set("NQST")

# ─── ENCABEZADO CSV ───────────────────────────────────────────────────────────
# NUEVO (v4): Taxon_ID, T_opt_C, Organism_Resolved — permiten agrupar por
# organismo real (GroupKFold / Leave-One-Organism-Out) y validar T_opt_C
# de forma continua, en vez de depender solo del binario Cold/Warm.
HEADER = (
    ['Protein_ID', 'Organism_Source', 'Taxon_ID', 'T_opt_C', 'Organism_Resolved',
     'EC_Class', 'Thermal_Class',
     'Length', 'Molecular_Weight', 'GRAVY', 'Instability_Index',
     'Aromaticity', 'Helix_Fraction', 'Turn_Fraction', 'Sheet_Fraction',
     'IVYWREL_Index', 'CvP_Bias', 'Flexibility_Ratio',
     ]
    + [f'AAC_{aa}' for aa in AMINO_ACIDS]
    + [f'DPC_{di}' for di in DIPEPTIDES]
)


def compute_thermal_features(seq: str) -> dict:
    slen = len(seq)
    ivywrel_count = sum(1 for aa in seq if aa in IVYWREL_SET)
    ivywrel_index = ivywrel_count / slen
    charged_count = sum(1 for aa in seq if aa in CHARGED_SET)
    polar_count   = sum(1 for aa in seq if aa in POLAR_SET)
    cvp_bias      = (charged_count - polar_count) / slen
    gly_count     = seq.count('G')
    ser_count     = seq.count('S')
    pro_count     = seq.count('P')
    flex_ratio    = (gly_count + ser_count) / (pro_count + 0.001)
    return {
        'IVYWREL_Index':     ivywrel_index,
        'CvP_Bias':          cvp_bias,
        'Flexibility_Ratio': flex_ratio,
    }


def extract_features(seq: str, record_id: str, header: str, taxa_index: dict,
                      ec_class: str, thermal_class: int, fallback_label: str) -> dict | None:
    """Extrae features + resuelve el organismo REAL desde el header FASTA (OS=/OX=)."""
    seq = seq.upper()
    if not set(seq).issubset(VALID_AA) or len(seq) < 20:
        return None

    try:
        analysis = ProteinAnalysis(seq)
        ss       = analysis.secondary_structure_fraction()
        aac      = analysis.amino_acids_percent
        seq_len  = len(seq)

        org_info = resolve_organism(header, taxa_index, thermal_class, fallback_label)

        feat = {
            'Protein_ID':        record_id,
            'Organism_Source':   org_info['Organism_Source'],
            'Taxon_ID':          org_info['Taxon_ID'],
            'T_opt_C':           org_info['T_opt_C'],
            'Organism_Resolved': org_info['Organism_Resolved'],
            'EC_Class':          ec_class,
            'Thermal_Class':     thermal_class,
            'Length':            seq_len,
            'Molecular_Weight':  analysis.molecular_weight(),
            'GRAVY':             analysis.gravy(),
            'Instability_Index': analysis.instability_index(),
            'Aromaticity':       analysis.aromaticity(),
            'Helix_Fraction':    ss[0],
            'Turn_Fraction':     ss[1],
            'Sheet_Fraction':    ss[2],
        }
        feat.update(compute_thermal_features(seq))
        feat['_label_mismatch'] = org_info['Label_Mismatch']  # se usa solo para el reporte, no va al CSV final

        for aa in AMINO_ACIDS:
            feat[f'AAC_{aa}'] = aac[aa]
        for di in DIPEPTIDES:
            feat[f'DPC_{di}'] = seq.count(di) / (seq_len - 1)

        return feat

    except Exception:
        return None


def main(force: bool = False):
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

    taxa_index = load_taxa_config(TAXA_CONFIG)

    print("\n" + "=" * 70)
    print("  PsychroScan — Extracción de Features v4")
    print("  NUEVO: Organism_Source real (antes: bug guardaba EC_Class)")
    print("=" * 70)
    print(f"  Input       : {RAW_DIR}/")
    print(f"  Output      : {OUT_FILE}")
    print(f"  Taxa config : {TAXA_CONFIG} ({len(taxa_index)} organismos curados)")
    print(f"  Archivos FASTA: {len(fastas)}")
    print("=" * 70 + "\n")

    total_valid, total_skipped   = 0, 0
    total_unresolved, total_mism = 0, 0
    class_counts = {0: 0, 1: 0}
    organisms_seen = set()

    with open(OUT_FILE, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HEADER, extrasaction='ignore')
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
            batch, file_valid, file_skip, file_mismatch, file_unresolved = [], 0, 0, 0, 0

            for record in SeqIO.parse(filepath, 'fasta'):
                feat = extract_features(
                    str(record.seq), record.id, record.description,
                    taxa_index, ec_class_name, thermal_class, base,
                )
                if feat is None:
                    file_skip += 1
                    continue

                if feat.pop('_label_mismatch'):
                    file_mismatch += 1
                if not feat['Organism_Resolved']:
                    file_unresolved += 1
                organisms_seen.add(feat['Organism_Source'])

                batch.append(feat)
                file_valid += 1
                if len(batch) >= BATCH_SIZE:
                    writer.writerows(batch)
                    batch = []

            if batch:
                writer.writerows(batch)

            total_valid      += file_valid
            total_skipped     += file_skip
            total_mism        += file_mismatch
            total_unresolved  += file_unresolved
            class_counts[thermal_class] += file_valid

            warn = f"  ⚠️  {file_mismatch} con Label_Mismatch (T_opt_C contradice Cold/Warm del archivo)" if file_mismatch else ""
            unres = f"  ⚠️  {file_unresolved} sin cruce a taxa_list (organismo no identificado)" if file_unresolved else ""
            print(f"  ✅ {file_valid:,} válidas  |  {file_skip} omitidas{warn}{unres}\n")

    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    print("=" * 70)
    print("  RESUMEN FINAL")
    print("=" * 70)
    print(f"  Total válidas       : {total_valid:,}")
    print(f"  Total omitidas      : {total_skipped:,}")
    print(f"  Organismos distintos: {len(organisms_seen)}")
    print(f"  CSV                 : {size_mb:.2f} MB  →  {OUT_FILE}")
    print(f"\n  ❄️  Cold (0) : {class_counts[0]:,} proteínas")
    print(f"  🌱  Warm (1) : {class_counts[1]:,} proteínas")

    if total_unresolved:
        print(f"\n  ⚠️  {total_unresolved:,} secuencias sin taxon_id resoluble contra "
              f"taxa_list.json.")
        print(f"     Revisa manualmente antes de usarlas para group-split (05_train_model.py).")
    if total_mism:
        print(f"\n  🚨 {total_mism:,} secuencias con Label_Mismatch: el archivo las etiqueta "
              f"Cold/Warm pero el T_opt_C real en taxa_list.json indica lo contrario.")
        print(f"     Esto es una señal de posible error de curación — revísalas antes de entrenar.")

    print(f"\n  Organismos únicos detectados (para group-split en 05):")
    for org in sorted(organisms_seen):
        print(f"    · {org}")

    print("\n  ✅ Siguiente paso → 05_train_model.py (ahora con split por organismo)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PsychroScan Feature Extraction v4")
    parser.add_argument('--force', action='store_true', help='Sobreescribir sin preguntar')
    args = parser.parse_args()
    main(force=args.force)
