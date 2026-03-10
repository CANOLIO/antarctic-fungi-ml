import os
import pandas as pd
import lightgbm as lgb
import joblib
import requests
import time

# Rutas
DATA_FILE = os.path.join("data", "processed", "dataset_features.csv")
MODELS_DIR = os.path.join("results", "models")
RESULTS_DIR = os.path.join("results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def fetch_uniprot_data(uniprot_id):
    """Consulta la API de UniProt para extraer metadatos biológicos críticos"""
    # El ID viene como 'tr|A0A1Y2G503|A0A1Y2G503_9BASI', necesitamos solo 'A0A1Y2G503'
    try:
        accession = uniprot_id.split('|')[1]
    except IndexError:
        accession = uniprot_id

    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    
    # Pausa de cortesía para no saturar la API
    time.sleep(0.5)
    
    response = requests.get(url)
    if response.status_code != 200:
        return "Error API", "Desconocida", "Ninguno"
    
    data = response.json()
    
    # 1. Extraer Nombre de la Proteína
    protein_name = "Proteína no caracterizada (Hipotética)"
    try:
        # Intenta sacar el nombre recomendado (Revisado)
        protein_name = data['proteinDescription']['recommendedName']['fullName']['value']
    except KeyError:
        try:
            # Si no, saca el nombre subido por el autor (TrEMBL)
            protein_name = data['proteinDescription']['submissionNames'][0]['fullName']['value']
        except (KeyError, IndexError):
            pass
            
    # 2. Extraer Localización Celular
    location = "No descrita"
    for comment in data.get('comments', []):
        if comment['commentType'] == 'SUBCELLULAR LOCATION':
            locations = [loc['location']['value'] for loc in comment.get('subcellularLocations', [])]
            location = ", ".join(locations)
            break
            
    # 3. Extraer Dominios Funcionales (Pfam)
    pfam_domains = []
    for xref in data.get('crossReferences', []):
        if xref['database'] == 'Pfam':
            # Extrae el nombre del dominio
            domain_name = next((p['value'] for p in xref.get('properties', []) if p['key'] == 'EntryName'), xref['id'])
            pfam_domains.append(domain_name)
            
    pfam_str = ", ".join(pfam_domains) if pfam_domains else "Sin dominios Pfam"
    
    return protein_name, location, pfam_str

def annotate_top_candidates():
    print("--- Fase Final: Anotación Biológica del Top 15 ---\n")
    
    print("Cargando modelo y base de datos...")
    df = pd.read_csv(DATA_FILE)
    model = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    
    X = df.drop(columns=['Protein_ID', 'Organism', 'Niche_Label']).astype('float32')
    
    print("Extrayendo firmas térmicas del proteoma global...")
    probs = model.predict_proba(X)[:, 0] # Probabilidad de ser Psicrófilo (Clase 0)
    
    # Reconstruimos el DataFrame con predicciones
    results_df = pd.DataFrame({
        'Protein_ID': df['Protein_ID'],
        'Organism': df['Organism'],
        'True_Class': df['Niche_Label'],
        'Probability': probs
    })
    
    # Filtramos solo los psicrófilos verdaderos y sacamos el Top 15
    psychrophiles = results_df[results_df['True_Class'] == 0]
    top_15 = psychrophiles.sort_values(by='Probability', ascending=False).head(15)
    
    print("\nConectando con la base de datos de UniProt para minería de datos...")
    print("-" * 60)
    
    annotations = []
    for idx, row in top_15.iterrows():
        prot_id = row['Protein_ID']
        prob = row['Probability'] * 100
        org = row['Organism']
        
        print(f"Consultando: {prot_id} ({prob:.2f}%) ...", end=" ")
        name, loc, pfam = fetch_uniprot_data(prot_id)
        print("✅ OK")
        
        annotations.append({
            'Organism': org,
            'Protein_ID': prot_id,
            'Prob_Frio': f"{prob:.2f}%",
            'Protein_Name': name,
            'Cellular_Location': loc,
            'Pfam_Domains': pfam
        })
        
    # Guardar reporte
    report_df = pd.DataFrame(annotations)
    out_csv = os.path.join(RESULTS_DIR, "top15_bioprospecting_report.csv")
    report_df.to_csv(out_csv, index=False)
    
    print("\n" + "="*80)
    print("🔬 REPORTE DE BIOPROSPECCIÓN INDUSTRIAL (Vista Previa)")
    print("="*80)
    for index, row in report_df.iterrows():
        print(f"[{index+1}] {row['Organism']} | P(Frío): {row['Prob_Frio']}")
        print(f"    ID: {row['Protein_ID']}")
        print(f"    Nombre: {row['Protein_Name']}")
        print(f"    Ubicación: {row['Cellular_Location']}")
        print(f"    Dominios: {row['Pfam_Domains']}")
        print("-" * 80)
        
    print(f"\n💾 Reporte completo exportado a: {out_csv}")

if __name__ == "__main__":
    annotate_top_candidates()