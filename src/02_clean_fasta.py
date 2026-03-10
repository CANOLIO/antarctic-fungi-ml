import os
from Bio import SeqIO

# 1. Configurar rutas
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 2. Definir los 20 aminoácidos estándar (alfabeto puro)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_fasta_file(filename):
    input_path = os.path.join(RAW_DIR, filename)
    output_path = os.path.join(PROCESSED_DIR, filename.replace(".fasta", "_cleaned.fasta"))
    
    valid_records = []
    seen_sequences = set() # Para eliminar duplicados exactos
    
    total_count = 0
    
    print(f"Procesando: {filename}...")
    
    # Analizar el archivo FASTA
    for record in SeqIO.parse(input_path, "fasta"):
        total_count += 1
        sequence = str(record.seq).upper()
        
        # Regla 1: Filtro de Longitud (50 a 1000 aa)
        if not (50 <= len(sequence) <= 1000):
            continue
            
        # Regla 2: Filtro de Pureza (Sin aminoácidos ambiguos)
        if not set(sequence).issubset(VALID_AA):
            continue
            
        # Regla 3: Filtro de Duplicados
        if sequence in seen_sequences:
            continue
            
        # Si pasa todas las reglas, lo guardamos
        seen_sequences.add(sequence)
        valid_records.append(record)
        
    # Guardar el archivo limpio
    SeqIO.write(valid_records, output_path, "fasta")
    
    # Reporte de métricas
    retention_rate = (len(valid_records) / total_count) * 100 if total_count > 0 else 0
    print(f"  -> Originales: {total_count}")
    print(f"  -> Limpias:    {len(valid_records)} ({retention_rate:.1f}% retenidas)\n")

if __name__ == "__main__":
    print("--- Iniciando Limpieza de Datos (Fungi-Clean) ---\n")
    
    # Buscar todos los archivos .fasta en la carpeta cruda
    for file in os.listdir(RAW_DIR):
        if file.endswith(".fasta"):
            clean_fasta_file(file)
            
    print("--- Limpieza Completada. Revisa data/processed ---")