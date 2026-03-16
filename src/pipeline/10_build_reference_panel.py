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
from io import StringIO

# ─── RUTAS ────────────────────────────────────────────────────────────────────
PANEL_DIR     = os.path.join("data", "reference_panel")
PANEL_FILE    = os.path.join(PANEL_DIR, "reference_profiles.csv")
PARTIAL_FILE  = os.path.join(PANEL_DIR, "reference_profiles_partial.csv")
os.makedirs(PANEL_DIR, exist_ok=True)

# ─── CONSTANTES ───────────────────────────────────────────────────────────────
AMINO_ACIDS  = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES   = [''.join(p) for p in iproduct(AMINO_ACIDS, repeat=2)]
VALID_AA     = set(AMINO_ACIDS)
IVYWREL_SET  = set("IVYWREL")
CHARGED_SET  = set("RKDE")
POLAR_SET    = set("NQST")
MAX_SEQS     = 150   # v2: aumentado de 30 → perfiles más robustos
MIN_SEQS     = 10    # mínimo para incluir organismo en el panel
SLEEP        = 0.40

# ─── LISTA DE ORGANISMOS ──────────────────────────────────────────────────────
# Thermal_Class: 0 = psicrófilo/psicrotrofo, 1 = mesófilo
ORGANISMS = [
    # ── PSICRÓFILOS / PSICROTROFOS ────────────────────────────────────────────
    ("Psychrobacter arcticus",          0, "Psychrobacter_arcticus"),
    ("Psychrobacter cryohalolentis",    0, "Psychrobacter_cryohalolentis"),
    ("Psychromonas ingrahamii",         0, "Psychromonas_ingrahamii"),
    ("Colwellia psychrerythraea",       0, "Colwellia_psychrerythraea"),
    ("Shewanella frigidimarina",        0, "Shewanella_frigidimarina"),
    ("Pseudoalteromonas haloplanktis",  0, "Pseudoalteromonas_haloplanktis"),
    ("Marinomonas primoryensis",        0, "Marinomonas_primoryensis"),
    ("Polaribacter irgensii",           0, "Polaribacter_irgensii"),
    ("Algoriphagus machipongonensis",   0, "Algoriphagus_machipongonensis"),
    ("Photobacterium profundum",        0, "Photobacterium_profundum"),
    ("Moritella marina",                0, "Moritella_marina"),
    ("Rhodococcus erythropolis",        0, "Rhodococcus_erythropolis"),
    ("Arthrobacter psychrolactophilus", 0, "Arthrobacter_psychrolactophilus"),
    ("Flavobacterium psychrophilum",    0, "Flavobacterium_psychrophilum"),
    ("Cryobacterium psychrotolerans",   0, "Cryobacterium_psychrotolerans"),
    ("Planococcus halocryophilus",      0, "Planococcus_halocryophilus"),
    ("Carnobacterium maltaromaticum",   0, "Carnobacterium_maltaromaticum"),
    ("Leucosporidium scottii",          0, "Leucosporidium_scottii"),
    ("Leucosporidium creatinivorum",    0, "Leucosporidium_creatinivorum"),
    ("Glaciozyma antarctica",           0, "Glaciozyma_antarctica"),
    ("Cryomyces antarcticus",           0, "Cryomyces_antarcticus"),
    ("Dioszegia hungarica",             0, "Dioszegia_hungarica"),
    ("Mrakia blollopis",                0, "Mrakia_blollopis"),
    ("Amycolatopsis antarctica",        0, "Amycolatopsis_antarctica"),
    ("Streptomyces cryophilus",         0, "Streptomyces_cryophilus"),
    # ── MESÓFILOS ─────────────────────────────────────────────────────────────
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
    retry = Retry(total=4, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def fetch_sequences(organism_name, session, max_seqs):
    """Descarga secuencias de UniProt. Intenta revisadas primero, luego todas."""
    url = "https://rest.uniprot.org/uniprotkb/stream"
    for reviewed in ["true", None]:
        query = f'organism_name:"{organism_name}"'
        if reviewed:
            query += " AND reviewed:true"
        params = {"query": query, "format": "fasta", "size": max_seqs}
        try:
            r = session.get(url, params=params, timeout=45)
            if r.status_code == 200 and r.text.strip():
                seqs = r.text.count('>')
                if seqs >= MIN_SEQS:
                    return r.text
        except Exception:
            pass
    return ""


def extract_features_from_text(fasta_text):
    """Extrae features de texto FASTA. Retorna DataFrame con una fila por proteína."""
    rows = []
    for record in SeqIO.parse(StringIO(fasta_text), 'fasta'):
        seq = str(record.seq).upper()
        if not set(seq).issubset(VALID_AA) or len(seq) < 20:
            continue
        try:
            analysis = ProteinAnalysis(seq)
            ss   = analysis.secondary_structure_fraction()
            aac  = analysis.amino_acids_percent
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
            rows.append(feat)
        except Exception:
            continue
    return pd.DataFrame(rows)


def build_panel(force=False, max_seqs=MAX_SEQS):
    # ── Checkpoint: cargar progreso parcial si existe ──────────────────────────
    completed_labels = set()
    partial_rows     = []
    if os.path.exists(PARTIAL_FILE) and not force:
        partial_df = pd.read_csv(PARTIAL_FILE)
        completed_labels = set(partial_df['Organism_Label'].tolist())
        partial_rows = partial_df.to_dict('records')
        print(f"  Retomando desde checkpoint: {len(completed_labels)} organismos ya procesados.")

    if os.path.exists(PANEL_FILE) and not force:
        print(f"  Panel completo ya existe: {PANEL_FILE}")
        print(f"  Usa --force para regenerar desde cero.")
        df = pd.read_csv(PANEL_FILE)
        print(f"  {len(df)} organismos en el panel.")
        return

    if force and os.path.exists(PARTIAL_FILE):
        os.remove(PARTIAL_FILE)
        completed_labels = set()
        partial_rows     = []

    print("\n" + "=" * 70)
    print("  PsychroScan — Panel de Referencia Taxonómica (v2)")
    print("=" * 70)
    print(f"  Organismos    : {len(ORGANISMS)}")
    print(f"  Seqs/organismo: {max_seqs}  (MIN={MIN_SEQS} para incluir)")
    print(f"  Output        : {PANEL_FILE}\n")

    session = get_session()
    rows    = list(partial_rows)

    pending = [(n, tc, lbl) for n, tc, lbl in ORGANISMS
               if lbl not in completed_labels]
    print(f"  Pendientes: {len(pending)} / {len(ORGANISMS)}\n")

    for i, (name, thermal_class, label) in enumerate(pending, 1):
        idx_global = len(rows) + 1
        print(f"  [{idx_global:02d}/{len(ORGANISMS)}] {name} ...", end=" ", flush=True)

        fasta_text = fetch_sequences(name, session, max_seqs)
        time.sleep(SLEEP)

        if not fasta_text:
            print(f"⚠️  sin secuencias suficientes (min={MIN_SEQS}), omitido")
            continue

        df_feat = extract_features_from_text(fasta_text)
        if len(df_feat) < MIN_SEQS:
            print(f"⚠️  solo {len(df_feat)} proteínas válidas < {MIN_SEQS}, omitido")
            continue

        profile = df_feat.mean().to_dict()
        profile['Organism_Name']  = name
        profile['Organism_Label'] = label
        profile['Thermal_Class']  = thermal_class
        profile['N_sequences']    = len(df_feat)
        rows.append(profile)
        print(f"✅ {len(df_feat)} seqs  (perfil basado en {len(df_feat)} proteínas)")

        # Guardar checkpoint parcial
        meta_cols = ['Organism_Name', 'Organism_Label', 'Thermal_Class', 'N_sequences']
        partial_df = pd.DataFrame(rows)
        feat_cols  = [c for c in partial_df.columns if c not in meta_cols]
        partial_df[meta_cols + feat_cols].to_csv(PARTIAL_FILE, index=False)

    # ── Panel final ────────────────────────────────────────────────────────────
    panel_df  = pd.DataFrame(rows)
    meta_cols = ['Organism_Name', 'Organism_Label', 'Thermal_Class', 'N_sequences']
    feat_cols = [c for c in panel_df.columns if c not in meta_cols]
    panel_df  = panel_df[meta_cols + feat_cols]
    panel_df.to_csv(PANEL_FILE, index=False)

    # Limpiar checkpoint
    if os.path.exists(PARTIAL_FILE):
        os.remove(PARTIAL_FILE)

    n_cold = (panel_df['Thermal_Class'] == 0).sum()
    n_warm = (panel_df['Thermal_Class'] == 1).sum()
    mean_seqs = panel_df['N_sequences'].mean()

    print(f"\n{'='*70}")
    print(f"  ✅ Panel guardado: {PANEL_FILE}")
    print(f"     ❄️  Psicrófilos  : {n_cold}")
    print(f"     🌱  Mesófilos    : {n_warm}")
    print(f"     Total            : {len(panel_df)} organismos")
    print(f"     Seqs/organismo   : {mean_seqs:.0f} promedio")
    print(f"\n  El script 09 usará este panel automáticamente.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Re-genera desde cero ignorando checkpoint')
    parser.add_argument('--max-seqs', type=int, default=MAX_SEQS,
                        help=f'Secuencias por organismo (default: {MAX_SEQS})')
    args = parser.parse_args()
    build_panel(force=args.force, max_seqs=args.max_seqs)