import os
import pandas as pd
import numpy as np
import joblib
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from itertools import product
import warnings
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler

# Ocultar advertencias molestas
warnings.filterwarnings("ignore")

# Rutas
NEW_GENOMES_DIR = os.path.join("data", "new_genomes")
MODELS_DIR = os.path.join("results", "models")
DATA_FILE = os.path.join("data", "processed", "dataset_features.csv")
os.makedirs(NEW_GENOMES_DIR, exist_ok=True)

# Reconstruir espacio de características
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = [''.join(pair) for pair in product(AMINO_ACIDS, repeat=2)]

def extract_features_from_fasta(filepath):
    valid_aa_set = set(AMINO_ACIDS)
    features_list = []
    protein_ids = []
    
    print(f"🔬 Extrayendo topología 3D y composición de: {os.path.basename(filepath)}...")
    for record in SeqIO.parse(filepath, "fasta"):
        seq = str(record.seq).upper()
        if not set(seq).issubset(valid_aa_set) or len(seq) < 20:
            continue
            
        analysis = ProteinAnalysis(seq)
        sec_struct = analysis.secondary_structure_fraction()
        aac = analysis.amino_acids_percent
        seq_len = len(seq)
        
        feat = {
            'Length': seq_len,
            'Molecular_Weight': analysis.molecular_weight(),
            'GRAVY': analysis.gravy(),
            'Instability_Index': analysis.instability_index(),
            'Aromaticity': analysis.aromaticity(),
            'Helix_Fraction': sec_struct[0],
            'Turn_Fraction': sec_struct[1],
            'Sheet_Fraction': sec_struct[2]
        }
        for aa in AMINO_ACIDS:
            feat[f'AAC_{aa}'] = aac[aa]
        for di in DIPEPTIDES:
            feat[f'DPC_{di}'] = seq.count(di) / (seq_len - 1)
            
        features_list.append(feat)
        protein_ids.append(record.id)
        
    return pd.DataFrame(features_list), protein_ids

def analyze_and_explain_genome():
    print("--- 🚀 MOTOR DE BIOPROSPECCIÓN EXPLICABLE (PPI + Dendrograma) ---\n")
    
    fasta_files = [f for f in os.listdir(NEW_GENOMES_DIR) if f.endswith('.fasta')]
    if not fasta_files:
        print("❌ No hay genomas nuevos para analizar.")
        return
        
    print("Cargando modelo LightGBM y base de datos de referencia...")
    model = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    original_df = pd.read_csv(DATA_FILE)
    original_profiles = original_df.drop(columns=['Protein_ID', 'Niche_Label']).groupby('Organism').mean()
    
    for fasta_file in fasta_files:
        organism_name = fasta_file.replace('.fasta', '')
        filepath = os.path.join(NEW_GENOMES_DIR, fasta_file)
        
        X_new, prot_ids = extract_features_from_fasta(filepath)
        if len(X_new) == 0: continue
            
        print("🧠 Calculando firmas termodinámicas y PPI...")
        probs = model.predict_proba(X_new.astype(np.float32))[:, 0]
        
        df_results = pd.DataFrame({'Protein_ID': prot_ids, 'Cold_Probability': probs})
        
        # --- 1. EL ÍNDICE PPI (Psychrophilic Proteome Index) ---
        top100_mean = df_results.nlargest(100, 'Cold_Probability')['Cold_Probability'].mean()
        pct_above_90 = (df_results['Cold_Probability'] >= 0.90).mean()
        
        flexibility_score = (X_new['AAC_G'].mean() + X_new['AAC_S'].mean()) / (X_new['AAC_P'].mean() + 0.001)
        norm_flexibility = min(flexibility_score / 8.0, 1.0) 
        
        ppi_score = (0.5 * top100_mean + 0.3 * pct_above_90 + 0.2 * norm_flexibility) * 100
        veredicto = "❄️ EXTREMÓFILO" if ppi_score > 40 else ("🌥️ TOLERANTE AL FRÍO" if ppi_score > 20 else "🌱 MESÓFILO")
        
        print("\n" + "="*80)
        print(f"🧬 REPORTE ECOLÓGICO EXPLICABLE: {organism_name.upper()}")
        print("="*80)
        print(f"Veredicto Ecológico        : {veredicto}")
        print(f"Psychrophilic Proteome Idx : {ppi_score:.2f} / 100 (PPI)")
        print(f"  ├─ Media Top-100 Enzimas : {top100_mean*100:.1f}%")
        print(f"  ├─ Fracción Ultra-Fría   : {pct_above_90*100:.2f}% del proteoma")
        print(f"  └─ Ratio de Flexibilidad : {flexibility_score:.2f} (Gly+Ser / Pro)")
        
        # --- 2. INTERPRETABILIDAD (Por qué las top 5 son frías) ---
        print("\n🔬 TOP 5 ENZIMAS Y SU FIRMA BIOQUÍMICA:")
        print("-" * 80)
        
        top_5_idx = df_results.nlargest(5, 'Cold_Probability').index
        
        for idx in top_5_idx:
            prot_id = prot_ids[idx]
            prob = probs[idx] * 100
            
            gly = X_new.iloc[idx]['AAC_G'] 
            pro = X_new.iloc[idx]['AAC_P'] 
            helix = X_new.iloc[idx]['Helix_Fraction'] * 100
            
            print(f"ID: {prot_id:<20} | Probabilidad: {prob:>6.2f}%")
            print(f"    ├─ Glicina (Bisagra)  : {gly:.1f}%")
            print(f"    ├─ Prolina (Rigidez)  : {pro:.1f}%")
            print(f"    └─ Propensión Hélice  : {helix:.1f}%")
        print("="*80)

        # --- 3. DENDROGRAMA CONTEXTUAL ---
        print(f"\n🌲 Generando mapa de posicionamiento ecológico...")
        
        new_profile = X_new.mean().to_frame().T
        new_profile.index = [f"⭐ {organism_name.upper()} (CANDIDATO NUEVO)"]
        
        combined_profiles = pd.concat([original_profiles, new_profile])
        
        scaler = StandardScaler()
        scaled_profiles = scaler.fit_transform(combined_profiles)
        Z = hierarchy.linkage(scaled_profiles, method='ward')
        
        plt.figure(figsize=(12, 10))
        plt.title(f"Posicionamiento Proteómico: {organism_name.upper()} vs Base de Referencia", fontsize=15, fontweight='bold')
        plt.xlabel("Distancia Bioquímica (Convergencia Adaptativa)")
        
        hierarchy.dendrogram(Z, labels=combined_profiles.index.tolist(), orientation='left', leaf_font_size=11)
        
        tree_path = os.path.join(NEW_GENOMES_DIR, f"{organism_name}_context_tree_PPI.png")
        plt.savefig(tree_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Mapa guardado en: {tree_path}\n")

if __name__ == "__main__":
    analyze_and_explain_genome()