import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

DATA_FILE = os.path.join("data", "processed", "dataset_features.csv")
MODELS_DIR = os.path.join("results", "models")
FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_publishable_figures():
    print("--- Generando Figuras de Validación Científica ---\n")
    
    # 1. Cargar Datos y Modelo
    print("Cargando datos y modelo...")
    df = pd.read_csv(DATA_FILE)
    model = joblib.load(os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))
    
    X = df.drop(columns=['Protein_ID', 'Organism', 'Niche_Label']).astype('float32')
    y = df['Niche_Label']
    
    # Probabilidades (Invertimos y para que Psicrófilo sea la clase positiva 1 en ROC)
    probs_psicrofilo = model.predict_proba(X)[:, 0]
    y_roc = 1 - y 

    # ---------------------------------------------------------
    # FIGURA A: CURVA ROC Y AUC
    # ---------------------------------------------------------
    print("Generando Curva ROC...")
    fpr, tpr, thresholds = roc_curve(y_roc, probs_psicrofilo)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR - Sensibilidad)')
    plt.title('Rendimiento del Clasificador de Enzimas Frías')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, '08A_ROC_Curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # FIGURA B: PCA DEL ESPACIO PROTEÓMICO
    # ---------------------------------------------------------
    print("Calculando PCA del espacio proteómico (esto tomará unos segundos)...")
    # Para que el PCA sea legible y no colapse la RAM, tomamos una muestra representativa
    df_pca = df.sample(n=min(30000, len(df)), random_state=42)
    X_pca = df_pca.drop(columns=['Protein_ID', 'Organism', 'Niche_Label']).astype('float32')
    y_pca = df_pca['Niche_Label']
    
    # Estandarizar antes de PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                          c=y_pca, cmap='coolwarm', alpha=0.5, s=15, edgecolors='none')
    
    # Crear leyenda personalizada
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ['Psicrófilo (Frío)', 'Mesófilo (Templado)'], loc="upper right")
    
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('Análisis de Componentes Principales (PCA) del Espacio Proteómico')
    plt.savefig(os.path.join(FIGURES_DIR, '08B_PCA_Space.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # FIGURA C: ANÁLISIS DE FLEXIBILIDAD (Gly, Ser, Pro)
    # ---------------------------------------------------------
    print("Analizando firmas de aminoácidos (Gly, Ser, Pro)...")
    # Añadimos las probabilidades al DF original para aislar las Top psicrófilas
    df['Cold_Prob'] = probs_psicrofilo
    
    # Grupo 1: Las top enzimas frías predichas por la IA (>90% probabilidad)
    top_cold = df[df['Cold_Prob'] > 0.90].copy()
    top_cold['Grupo'] = 'Top Candidatas Frías'
    
    # Grupo 2: Mesófilos seguros (<10% probabilidad)
    mesophiles = df[(df['Niche_Label'] == 1) & (df['Cold_Prob'] < 0.10)].sample(n=len(top_cold), random_state=42).copy()
    mesophiles['Grupo'] = 'Mesófilos Base'
    
    comparison_df = pd.concat([top_cold, mesophiles])
    
    # Preparar datos para Boxplot
    aa_melted = pd.melt(comparison_df, id_vars=['Grupo'], value_vars=['AAC_G', 'AAC_S', 'AAC_P'], 
                        var_name='Aminoácido', value_name='Porcentaje')
    
    # Renombrar para que se vea bien en el gráfico
    aa_melted['Aminoácido'] = aa_melted['Aminoácido'].map({'AAC_G': 'Glicina (Flexibilidad)', 'AAC_S': 'Serina (Puentes H)', 'AAC_P': 'Prolina (Rigidez)'})
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Aminoácido', y='Porcentaje', hue='Grupo', data=aa_melted, palette=['#74b9ff', '#ff7675'], showfliers=False)
    plt.title('Firma Aminoacídica: Adaptación Térmica (Validación Biológica)')
    plt.ylabel('Frecuencia en la Secuencia')
    plt.xlabel('')
    plt.savefig(os.path.join(FIGURES_DIR, '08C_AA_Composition.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ ¡Validaciones terminadas!")
    print(f"📊 Revisa la carpeta '{FIGURES_DIR}' para ver tus nuevos gráficos listos para el paper.")

if __name__ == "__main__":
    generate_publishable_figures()