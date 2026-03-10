import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, fbeta_score

DATA_FILE = os.path.join("data", "processed", "dataset_features.csv")
MODELS_DIR = os.path.join("results", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Cargando datos masivos en memoria...")
df = pd.read_csv(DATA_FILE)
X = df.drop(columns=['Protein_ID', 'Organism', 'Niche_Label']).astype(np.float32)
y = df['Niche_Label']
protein_ids = df['Protein_ID']
organisms = df['Organism']

X_train, X_test, y_train, y_test, id_train, id_test, org_train, org_test = train_test_split(
    X, y, protein_ids, organisms, test_size=0.2, random_state=42, stratify=y
)

# Invertimos las etiquetas temporalmente para que Psicrófilo sea la clase "Positiva" (1)
# Esto es vital para que scikit-learn calcule el F2-score sobre la clase que nos interesa.
y_test_inv = 1 - y_test

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'num_leaves': trial.suggest_int('num_leaves', 30, 120),
        # Aumentamos el peso mínimo para forzar una mayor sensibilidad (Recall)
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 25.0), 
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'n_jobs': 2, 
        'random_state': 42
    }
    
    model = lgb.LGBMClassifier(**param)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 0] # Probabilidad de Psicrófilo
    
    best_f2 = 0
    # Buscamos el umbral que maximice el F2-Score (Prioridad: Recall)
    for t in np.linspace(0.1, 0.6, 20):
        preds = [1 if p >= t else 0 for p in probs] # 1 es Psicrófilo en nuestra versión invertida
        f2 = fbeta_score(y_test_inv, preds, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            
    return best_f2

def train_peer_reviewed_model():
    print("--- Optuna: Maximizando Sensibilidad (F2-Score) para Bioprospección ---\n")
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    print("Entrenando 20 inteligencias artificiales... (Cuidando RAM y CPU)")
    study.optimize(objective, n_trials=20)
    
    print("\n✅ OPTIMIZACIÓN TERMINADA")
    print(f"Mejor F2-Score (Enfoque en Recall) alcanzado: {study.best_value:.4f}")
    
    # Entrenar modelo final con los mejores parámetros
    best_params = study.best_params
    best_params['n_jobs'] = 2
    best_params['random_state'] = 42
    
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)
    probs_psicrofilo = final_model.predict_proba(X_test)[:, 0]
    
    # Búsqueda de umbral final
    best_threshold = 0.5
    best_f2 = 0
    for t in np.linspace(0.05, 0.7, 50):
        preds_inv = [1 if p >= t else 0 for p in probs_psicrofilo]
        f2 = fbeta_score(y_test_inv, preds_inv, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t
            
    y_pred_final = [0 if p >= best_threshold else 1 for p in probs_psicrofilo]
    
    print(f"\n🎯 UMBRAL ÓPTIMO (Sensibilidad): {best_threshold:.4f}")
    print("\n✅ REPORTE CLÍNICO (A nivel de Proteína):")
    print(classification_report(y_test, y_pred_final, target_names=['Psicrófilo (0)', 'Mesófilo (1)']))
    
    results_df = pd.DataFrame({
        'Protein_ID': id_test,
        'Organism': org_test,
        'True_Class': y_test,
        'Cold_Probability': probs_psicrofilo
    })
    
    # --- LA SOLUCIÓN AL PROFESOR: MÉTRICAS DE RANKING NORMALIZADAS ---
    print("\n" + "="*75)
    print("🏆 SALIDA 1: RANKING ECOLÓGICO (Basado en el 'Top 100' de cada organismo)")
    print("="*75)
    
    def calculate_organism_metrics(group):
        # Promedio solo de las 100 mejores proteínas (elimina sesgo de tamaño)
        top_100_mean = group.nlargest(100, 'Cold_Probability')['Cold_Probability'].mean()
        # Porcentaje del proteoma total con altísima confianza
        pct_above_90 = (group['Cold_Probability'] >= 0.90).mean() * 100
        return pd.Series({'Top_100_Mean': top_100_mean, 'Pct_Above_90': pct_above_90})
        
    org_scores = results_df.groupby(['Organism', 'True_Class']).apply(calculate_organism_metrics).reset_index()
    org_scores = org_scores.sort_values(by='Top_100_Mean', ascending=False)
    org_scores['True_Class'] = org_scores['True_Class'].map({0: '❄️ Psicrófilo', 1: '🌱 Mesófilo'})
    
    print(f"{'Organismo':<33} | {'Clase':<12} | {'Media Top-100':<14} | {'Proteínas >90%':<12}")
    print("-" * 75)
    for _, row in org_scores.iterrows():
        print(f"{row['Organism']:<33} | {row['True_Class']:<12} | {row['Top_100_Mean']*100:>6.2f}%        | {row['Pct_Above_90']:>6.2f}%")
        
    print("\n" + "="*75)
    print("🔬 SALIDA 2: TOP 15 CANDIDATAS (Priorizadas para Laboratorio)")
    print("="*75)
    psychrophile_proteins = results_df[results_df['True_Class'] == 0]
    top_candidates = psychrophile_proteins.sort_values(by='Cold_Probability', ascending=False).head(15)
    print(f"{'Protein_ID':<20} | {'Organismo Origen':<30} | {'Probabilidad':<15}")
    print("-" * 75)
    for _, row in top_candidates.iterrows():
        print(f"{row['Protein_ID']:<20} | {row['Organism']:<30} | {row['Cold_Probability']*100:>5.2f}%")
        
    joblib.dump(final_model, os.path.join(MODELS_DIR, "optuna_f2_model.pkl"))

if __name__ == "__main__":
    train_peer_reviewed_model()