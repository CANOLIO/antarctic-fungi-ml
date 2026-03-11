"""
PsychroScan — 10_build_reference_panel.py
==========================================
Construye el panel de referencia taxonómica para el dendrograma del script 09.

Usa la misma lista de organismos de 01b_fetch_brenda_coldenzymes.py para
que el dendrograma sea coherente con el modelo entrenado. Para cada organismo,
descarga hasta MAX_SEQS secuencias de UniProt, extrae features, y promedia.

El resultado es un CSV con una fila por organismo:
  data/reference_panel/reference_profiles.csv

Este archivo es estático — se genera una sola vez y el script 09 lo carga
automáticamente para el dendrograma de posicionamiento.

Uso:
    python 10_build_reference_panel.py
    python 10_build_reference_panel.py --force   # re-genera aunque ya exista
"""

import os
import sys
import time
import argparse
import requests
import numpy as np
import pandas as pd
from itertools import product as iproduct
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─── RUTAS ────────────────────────────────────────────────────────────────────
PANEL_DIR  = os.path.join("data", "reference_panel")
PANEL_FILE = os.path.join(PANEL_DIR, "reference_profiles.csv")
os.makedirs(PANEL_DIR, exist_ok=True)

# ─── CONSTANTES ───────────────────────────────────────────────────────────────
AMINO_ACIDS  = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES   = [''.join(p) for p in iproduct(AMINO_ACIDS, repeat=2)]
VALID_AA     = set(AMINO_ACIDS)
IVYWREL_SET  = set("IVYWREL")
CHARGED_SET  = set("RKDE")
POLAR_SET    = set("NQST")
MAX_SEQS     = 30    # Secuencias por organismo para promediar perfil
SLEEP        = 0.35  # segundos entre llamadas a UniProt

# ─── LISTA DE ORGANISMOS (espejo de 01b, con clase térmica explícita) ─────────
# Thermal_Class: 0 = psicrófilo/psicrotrofo, 1 = mesófilo
ORGANISMS = [
    # ── PSICRÓFILOS / PSICROTROFOS ──────────────────────────────────────────
    # Bacterias polares clásicas
    ("Psychrobacter arcticus",          0, "Psychrobacter_arcticus"),
    ("Psychrobacter cryohalolentis",    0, "Psychrobacter_cryohalolentis"),
    ("Psychromonas ingrahamii",         0, "Psychromonas_ingrahamii"),
    ("Colwellia psychrerythraea",       0, "Colwellia_psychrerythraea"),
    ("Shewanella frigidimarina",        0, "Shewanella_frigidimarina"),
    ("Pseudoalteromonas haloplanktis",  0, "Pseudoalteromonas_haloplanktis"),
    ("Marinomonas primoryensis",        0, "Marinomonas_primoryensis"),
    ("Polaribacter irgensii",           0, "Polaribacter_irgensii"),
    ("Algoriphagus machipongonensis",   0, "Algoriphagus_machipongonensis"),
    # Deep-sea
    ("Photobacterium profundum",        0, "Photobacterium_profundum"),
    ("Moritella marina",                0, "Moritella_marina"),
    # Suelo ártico / antártico
    ("Rhodococcus erythropolis",        0, "Rhodococcus_erythropolis"),
    ("Arthrobacter psychrolactophilus", 0, "Arthrobacter_psychrolactophilus"),
    ("Flavobacterium psychrophilum",    0, "Flavobacterium_psychrophilum"),
    ("Cryobacterium psychrotolerans",   0, "Cryobacterium_psychrotolerans"),
    ("Planococcus halocryophilus",      0, "Planococcus_halocryophilus"),
    ("Carnobacterium maltaromaticum",   0, "Carnobacterium_maltaromaticum"),
    # Levaduras / hongos fríos
    ("Leucosporidium scottii",          0, "Leucosporidium_scottii"),
    ("Leucosporidium creatinivorum",    0, "Leucosporidium_creatinivorum"),
    ("Glaciozyma antarctica",           0, "Glaciozyma_antarctica"),
    ("Cryomyces antarcticus",           0, "Cryomyces_antarcticus"),
    ("Dioszegia hungarica",             0, "Dioszegia_hungarica"),
    ("Mrakia blollopis",                0, "Mrakia_blollopis"),
    # Actinobacterias antárticas
    ("Amycolatopsis antarctica",        0, "Amycolatopsis_antarctica"),
    ("Streptomyces cryophilus",         0, "Streptomyces_cryophilus"),

    # ── MESÓFILOS (controles) ────────────────────────────────────────────────
    ("Saccharomyces cerevisiae",        1, "Saccharomyces_cerevisiae"),
    ("Schizosaccharomyces pombe",       1, "Schizosaccharomyces_pombe"),
    ("Candida albicans",                1, "Candida_albicans"),
    ("Aspergillus niger",               1, "Aspergillus_niger"),
    ("Penicillium rubens",              1, "Penicillium_rubens"),
    ("Neurospora crassa",               1, "Neurospora_crassa"),
    ("Trichoderma reesei",              1, "Trichoderma_reesei"),
    ("Escherichia coli",                1, "Escherichia_coli"),
    ("Bacillus subtilis",               1, "Bacillus_subtilis"),
    ("Pseudomonas aeruginosa",          1, "Pseudomonas_aeruginosa"),
    ("Staphylococcus aureus",           1, "Staphylococcus_aureus"),
    ("Streptomyces griseus",            1, "Streptomyces_griseus"),
]


def get_session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=[429,500,502,503])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def fetch_sequences(organism_name, session, max_seqs=MAX_SEQS):
    """Descarga secuencias de proteínas revisadas de UniProt para un organismo."""
    query = f'organism_name:"{organism_name}" AND reviewed:true'
    url   = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query":  query,
        "format": "fasta",
        "size":   max_seqs,
    }
    try:
        r = session.get(url, params=params, timeout=30)
        if r.status_code != 200 or not r.text.strip():
            # Fallback: incluir no revisadas
            params["query"] = f'organism_name:"{organism_name}"'
            r = session.get(url, params=params, timeout=30)
        if r.status_code == 200 and r.text.strip():
            return r.text
    except Exception:
        pass
    return ""


def extract_features_from_fasta_text(fasta_text):
    """Extrae features de texto FASTA y devuelve DataFrame."""
    from io import StringIO
    features_list = []
    for record in SeqIO.parse(StringIO(fasta_text), 'fasta'):
        seq = str(record.seq).upper()
        if not set(seq).issubset(VALID_AA) or len(seq) < 20:
            continue
        try:
            analysis = ProteinAnalysis(seq)
            ss  = analysis.secondary_structure_fraction()
            aac = analysis.amino_acids_percent
            slen = len(seq)
            feat = {
                'Length':            slen,
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
            features_list.append(feat)
        except Exception:
            continue
    return pd.DataFrame(features_list)


def build_panel(force=False):
    if os.path.exists(PANEL_FILE) and not force:
        print(f"  Panel ya existe: {PANEL_FILE}")
        print(f"  Usa --force para regenerar.")
        df = pd.read_csv(PANEL_FILE)
        print(f"  {len(df)} organismos en el panel.")
        return

    print("\n" + "=" * 70)
    print("  PsychroScan — Construcción del Panel de Referencia Taxonómica")
    print("=" * 70)
    print(f"  Organismos a procesar : {len(ORGANISMS)}")
    print(f"  Secuencias por org.   : {MAX_SEQS}")
    print(f"  Output                : {PANEL_FILE}\n")

    session = get_session()
    rows = []

    for i, (name, thermal_class, label) in enumerate(ORGANISMS, 1):
        print(f"  [{i:02d}/{len(ORGANISMS)}] {name} ...", end=" ", flush=True)
        fasta_text = fetch_sequences(name, session)
        time.sleep(SLEEP)

        if not fasta_text:
            print("⚠️  sin secuencias, omitido")
            continue

        df_feat = extract_features_from_fasta_text(fasta_text)
        if len(df_feat) == 0:
            print("⚠️  sin features válidas, omitido")
            continue

        profile = df_feat.mean().to_dict()
        profile['Organism_Name']  = name
        profile['Organism_Label'] = label
        profile['Thermal_Class']  = thermal_class
        profile['N_sequences']    = len(df_feat)
        rows.append(profile)
        print(f"✅ {len(df_feat)} seqs")

    panel_df = pd.DataFrame(rows)
    # Reordenar: metadatos primero
    meta_cols = ['Organism_Name', 'Organism_Label', 'Thermal_Class', 'N_sequences']
    feat_cols = [c for c in panel_df.columns if c not in meta_cols]
    panel_df  = panel_df[meta_cols + feat_cols]
    panel_df.to_csv(PANEL_FILE, index=False)

    n_cold = (panel_df['Thermal_Class'] == 0).sum()
    n_warm = (panel_df['Thermal_Class'] == 1).sum()
    print(f"\n  ✅ Panel guardado: {PANEL_FILE}")
    print(f"     ❄️  Psicrófilos : {n_cold}")
    print(f"     🌱  Mesófilos   : {n_warm}")
    print(f"     Total           : {len(panel_df)} organismos")
    print(f"\n  Siguiente paso → el script 09 usará este panel automáticamente.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Re-genera el panel aunque ya exista')
    args = parser.parse_args()
    build_panel(force=args.force)