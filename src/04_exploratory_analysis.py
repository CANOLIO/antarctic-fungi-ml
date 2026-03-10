import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Rutas
DATA_FILE = os.path.join("data", "processed", "dataset_features.csv")
RESULTS_DIR = os.path.join("results", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_eda():
    print("--- Iniciando Análisis Exploratorio de Datos (EDA) ---\n")
    
    # Cargar los datos numéricos
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {DATA_FILE}")
        return

    # Mapear las etiquetas numéricas a texto para que el gráfico sea legible
    df['Nicho_Ecologico'] = df['Niche_Label'].map({0: 'Psicrófilo (Frío Extremo)', 1: 'Mesófilo (Templado)'})
    
    # Configurar estilo visual profesional
    sns.set_theme(style="whitegrid", palette="muted")
    
    print("🎨 Generando Gráfico 1: Proporción de Prolina...")
    plt.figure(figsize=(8, 6))
    # Boxplot para ver la mediana y los valores atípicos de la Prolina (rigidez)
    sns.boxplot(x='Nicho_Ecologico', y='AAC_P', data=df, hue='Nicho_Ecologico', legend=False)
    plt.title('Comparación de Prolina (AAC_P) por Nicho Ecológico', fontsize=14)
    plt.ylabel('Frecuencia de Prolina en la Proteína')
    plt.xlabel('')
    plt.savefig(os.path.join(RESULTS_DIR, '01_proline_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("🎨 Generando Gráfico 2: Punto Isoeléctrico (pI)...")
    plt.figure(figsize=(8, 6))
    # Violinplot para ver la distribución de la densidad del pI
    sns.violinplot(x='Nicho_Ecologico', y='Isoelectric_Point', data=df, inner="quartile", hue='Nicho_Ecologico', legend=False)
    plt.title('Distribución del Punto Isoeléctrico (pI)', fontsize=14)
    plt.ylabel('pH')
    plt.xlabel('')
    plt.savefig(os.path.join(RESULTS_DIR, '02_isoelectric_point.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("🎨 Generando Gráfico 3: Índice Hidropático (GRAVY)...")
    plt.figure(figsize=(8, 6))
    # KDE Plot (Gráfico de densidad) para ver cómo se agrupa la hidrofobicidad
    sns.kdeplot(data=df, x="GRAVY", hue="Nicho_Ecologico", fill=True, common_norm=False, alpha=0.5, linewidth=2)
    plt.title('Densidad de Hidrofobicidad Global (GRAVY)', fontsize=14)
    plt.xlabel('Índice GRAVY (Más alto = Más hidrofóbico)')
    plt.ylabel('Densidad')
    plt.savefig(os.path.join(RESULTS_DIR, '03_gravy_density.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ ¡Éxito! Gráficas generadas en la carpeta: {RESULTS_DIR}")
    print("Abre las imágenes para confirmar tus hipótesis bioquímicas.")

if __name__ == "__main__":
    run_eda()