import os
import requests
import time
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- DATASET CURADO MANUALMENTE (Taxon IDs Exactos) ---
ORGANISMS = {
    # ❄️ CLASE 0: Psicrófilos y Extremófilos
    "Glaciozyma_antarctica": "105987",
    "Pseudogymnoascus_destructans": "655981",
    "Leucosporidium_creatinivorum": "106004",
    "Leucosporidium_scottii": "5278",
    "Typhula_ishikariensis": "69361",
    "Mrakia_blollopis": "696254",
    "Cryomyces_antarcticus": "329879",
    "Thelebolus_microsporus": "319061",
    "Antarctomyces_psychrotrophicus": "89416",
    "Goffeauzyma_gastrica": "92955",
    "Dioszegia_hungarica": "4972",
    "Holtermanniella_wattica": "207911",
    "Phenoliferia_glacialis": "418497",  # ¡Bienvenido de vuelta!

    # 🌡️ CLASE 1: Mesófilos
    "Saccharomyces_cerevisiae": "4932",
    "Aspergillus_niger": "5061",
    "Neurospora_crassa": "5141",
    "Schizosaccharomyces_pombe": "4896",
    "Trichoderma_reesei": "51453",
    "Botrytis_cinerea": "40559",
    "Candida_albicans": "5476",
    "Cryptococcus_neoformans": "5207",
    "Ustilago_maydis": "5270",
    "Magnaporthe_oryzae": "318829",
    "Agaricus_bisporus": "5341",
    "Coprinopsis_cinerea": "5346",
    "Laccaria_bicolor": "29883",
    "Penicillium_rubens": "1108849"
}

RAW_DATA_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def create_robust_session():
    """Crea una sesión HTTP que reintenta automáticamente si el servidor falla"""
    session = requests.Session()
    retry = Retry(
        total=5, # Reintenta 5 veces
        backoff_factor=1, # Espera 1s, 2s, 4s... entre intentos
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def fetch_proteome(organism_name, taxon_id):
    file_path = os.path.join(RAW_DATA_DIR, f"{organism_name}.fasta")
    
    # Auditoría: Si el archivo existe pero está vacío o casi vacío (< 10 KB), lo borramos.
    if os.path.exists(file_path):
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb > 10:
            return f"⏭️ Omitido: {organism_name} ya existe y parece válido ({size_kb:.1f} KB)."
        else:
            os.remove(file_path)
            print(f"⚠️ {organism_name} estaba corrupto o vacío. Reintentando...")

    url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=(taxonomy_id:{taxon_id})"
    session = create_robust_session()
    
    try:
        # stream=True previene que la memoria colapse durante la descarga
        with session.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        
        # Auditoría post-descarga
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb < 10:
            os.remove(file_path)
            return f"❌ Fallo en {organism_name}: Archivo descargado estaba vacío."
            
        return f"✅ Descargado con éxito: {organism_name} ({size_kb:.1f} KB)"
    
    except Exception as e:
        return f"❌ Error crítico en {organism_name}: {e}"

def main():
    print(f"--- Iniciando Descarga Blindada ({len(ORGANISMS)} Organismos) ---")
    start_time = time.time()
    
    # Bajamos los "trabajadores" a 3 para no enfadar a UniProt
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_proteome, name, tax_id): name for name, tax_id in ORGANISMS.items()}
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
            
    print(f"\n⏱️ Tiempo total: {(time.time() - start_time)/60:.2f} minutos.")

if __name__ == "__main__":
    main()