import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler

# Rutas
DATA_FILE = os.path.join("data", "processed", "dataset_features.csv")
FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def build_proteomic_dendrogram():
    print("--- Generando Dendrograma Proteómico (Alignment-Free) ---\n")
    
    # Cargar datos
    df = pd.read_csv(DATA_FILE)
    
    # 1. Crear el perfil global por organismo (Promedio de sus características)
    print("Calculando firmas globales ecofisiológicas...")
    organism_profiles = df.drop(columns=['Protein_ID', 'Niche_Label']).groupby('Organism').mean()
    
    # 2. Normalizar
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(organism_profiles)
    
    # 3. Calcular distancias (Ward minimiza la varianza interna de los clados)
    print("Construyendo agrupamiento jerárquico...")
    Z = hierarchy.linkage(scaled_profiles, method='ward')
    
    # 4. Dibujar el Gráfico Blindado para Peer-Review
    plt.figure(figsize=(12, 9)) # Un poco más alto para que quepa la nota
    
    # Títulos precisos (Resolviendo la crítica del profesor)
    plt.title("Dendrograma de Similitud Proteómica Global", fontsize=16, fontweight='bold')
    plt.suptitle("Agrupamiento por Convergencia Adaptativa y Composición Bioquímica", y=0.92, fontsize=12, color='gray')
    plt.xlabel("Distancia Bioquímica (No puramente filogenética)")
    plt.ylabel("Organismos")
    
    # Dibujar árbol
    hierarchy.dendrogram(
        Z, 
        labels=organism_profiles.index.tolist(), 
        orientation='left',
        leaf_font_size=12,
        color_threshold=15 # Ajusta el color de las ramas principales
    )
    
    # EL TOQUE MAESTRO: Nota aclaratoria en la imagen para el jurado
    nota = (
        "Nota Metodológica:\n"
        "Este árbol es 'Alignment-Free' y agrupa taxones basados en el promedio de sus fenotipos\n"
        "estructurales (AAC, DPC, Sec. Struct.). Refleja similitud ecofisiológica y adaptativa, "
        "no ascendencia evolutiva estricta basada en ortólogos."
    )
    plt.figtext(0.5, -0.05, nota, ha="center", fontsize=10, 
                bbox={"facecolor":"#fff3cd", "edgecolor":"#ffeeba", "alpha":0.8, "pad":8})
    
    # Guardar
    tree_path = os.path.join(FIGURES_DIR, '06_proteomic_dendrogram.png')
    plt.savefig(tree_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ¡Dendrograma generado exitosamente!")
    print(f"🌲 Gráfico guardado en: {tree_path}")

if __name__ == "__main__":
    build_proteomic_dendrogram()