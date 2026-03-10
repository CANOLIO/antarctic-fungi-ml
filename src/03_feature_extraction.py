import os
import csv
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from itertools import product

# 1. Rutas
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Recuerda poner aquí tu lista completa y actualizada de organismos
LABELS = {
    # Psicrófilos (0)
    "Glaciozyma_antarctica": 0, "Pseudogymnoascus_destructans": 0,
    "Leucosporidium_creatinivorum": 0, "Leucosporidium_scottii": 0,
    "Antarctomyces_psychrotrophicus": 0, "Typhula_ishikariensis": 0,
    "Mrakia_blollopis": 0, "Cryomyces_antarcticus": 0,
    "Thelebolus_microsporus": 0, "Dioszegia_hungarica": 0, 
    "Goffeauzyma_gastrica": 0, "Holtermanniella_wattica": 0,
    
    # Mesófilos (1)
    "Saccharomyces_cerevisiae": 1, "Aspergillus_niger": 1,
    "Neurospora_crassa": 1, "Schizosaccharomyces_pombe": 1,
    "Trichoderma_reesei": 1, "Penicillium_rubens": 1,
    "Fusarium_oxysporum": 1, "Botrytis_cinerea": 1,
    "Candida_albicans": 1, "Cryptococcus_neoformans": 1,
    "Ustilago_maydis": 1, "Magnaporthe_oryzae": 1,
    "Agaricus_bisporus": 1, "Coprinopsis_cinerea": 1,
    "Laccaria_bicolor": 1
}

# Generar los 400 pares de dipéptidos
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = [''.join(pair) for pair in product(AMINO_ACIDS, repeat=2)]

def extract_3d_features_optimized():
    print("--- Iniciando Extracción Avanzada (Optimizada para RAM) ---\n")
    
    out_file = os.path.join(PROCESSED_DATA_DIR, "dataset_features.csv")
    valid_aa_set = set(AMINO_ACIDS)
    
    # Preparamos las columnas (Encabezado)
    header = ['Protein_ID', 'Organism', 'Niche_Label', 'Length', 'Molecular_Weight', 
              'GRAVY', 'Instability_Index', 'Aromaticity', 
              'Helix_Fraction', 'Turn_Fraction', 'Sheet_Fraction']
    header += [f'AAC_{aa}' for aa in AMINO_ACIDS]
    header += [f'DPC_{di}' for di in DIPEPTIDES]
    
    # Abrimos el archivo en modo escritura y lo vamos llenando en tiempo real
    with open(out_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        for filename in os.listdir(RAW_DATA_DIR):
            if not filename.endswith(".fasta"):
                continue
                
            organism_name = filename.replace(".fasta", "")
            if organism_name not in LABELS:
                continue
                
            niche_label = LABELS[organism_name]
            filepath = os.path.join(RAW_DATA_DIR, filename)
            
            print(f"Analizando proteoma: {organism_name}...")
            
            batch = []
            valid_count = 0
            
            for record in SeqIO.parse(filepath, "fasta"):
                seq = str(record.seq).upper()
                
                # Filtro de bioseguridad
                if not set(seq).issubset(valid_aa_set) or len(seq) < 20:
                    continue
                    
                analysis = ProteinAnalysis(seq)
                
                # --- CARACTERÍSTICAS GLOBALES ---
                features = {
                    'Protein_ID': record.id,
                    'Organism': organism_name,
                    'Niche_Label': niche_label,
                    'Length': len(seq),
                    'Molecular_Weight': analysis.molecular_weight(),
                    'GRAVY': analysis.gravy(),
                    'Instability_Index': analysis.instability_index(),
                    'Aromaticity': analysis.aromaticity()
                }
                
                # --- PROPENSIÓN ESTRUCTURAL ---
                sec_struct = analysis.secondary_structure_fraction()
                features['Helix_Fraction'] = sec_struct[0]
                features['Turn_Fraction'] = sec_struct[1]
                features['Sheet_Fraction'] = sec_struct[2]
                
                # --- COMPOSICIÓN AAC (Corrección de DeprecationWarning) ---
                aac = analysis.amino_acids_percent
                for aa in AMINO_ACIDS:
                    features[f'AAC_{aa}'] = aac[aa]
                    
                # --- COMPOSICIÓN DPC (Motivos 3D) ---
                seq_len = len(seq)
                for di in DIPEPTIDES:
                    features[f'DPC_{di}'] = seq.count(di) / (seq_len - 1)
                    
                batch.append(features)
                valid_count += 1
                
                # --- MAGIA DEL STREAMING: Vaciar RAM cada 5000 proteínas ---
                if len(batch) >= 5000:
                    writer.writerows(batch)
                    batch = [] # Se libera la RAM aquí
            
            # Guardar cualquier sobrante que quedó en el último lote
            if batch:
                writer.writerows(batch)
                
            print(f"  -> ✅ Procesadas y guardadas {valid_count} proteínas.")

    print(f"\n✅ ¡Éxito! Base de datos gigante creada de forma segura.")
    print(f"💾 Guardado en: {out_file}")

if __name__ == "__main__":
    extract_3d_features_optimized()