import numpy as np
import pandas as pd

FUNGI_GENERA = {
    'saccharomyces', 'schizosaccharomyces', 'candida', 'aspergillus', 'neurospora',
    'trichoderma', 'botrytis', 'ustilago', 'magnaporthe', 'yarrowia', 'rhodotorula',
    'leucosporidium', 'glaciozyma', 'mrakia', 'cryomyces', 'pseudogymnoascus',
    'thelebolus', 'phenoliferia', 'goffeauzyma', 'guehomyces', 'tausonia',
    'naganishia', 'geomyces', 'cladosporium', 'penicillium', 'geotrichum',
    'pyricularia', 'emericella', 'mycosarcoma', 'dioszegia', 'sungouiella',
    'friedmanniomyces', 'rachicladosporium', 'phaffia', 'debaryomyces'
}

class HierarchicalPsychroScan:
    """
    Arquitectura Jerárquica Condicionada por Dominio (Two-Stage Pipeline):
      - Etapa 1: Clasificador de Dominio (Bacteria vs Fungi)
      - Etapa 2A: Rama Bacteriana (Feature Selection bacteriano + Ensamble bacteriano + tau_b)
      - Etapa 2B: Rama Fúngica (Feature Selection fúngico + Ensamble fúngico + tau_f)
    
    Garantiza compatibilidad 100% con toda la suite downstream:
      predict_proba(X)[:, 0] -> P(Cold)
      predict_proba(X)[:, 1] -> P(Warm)
    """
    def __init__(self, domain_pipe, bact_branch, fungi_branch, tau_b=0.2800, tau_f=0.2300, bact_cols=None, fungi_cols=None):
        self.domain_pipe  = domain_pipe
        self.bact_branch  = bact_branch   # dict: {'sel': sel_b, 'lgb': m_l, 'rf': m_r, 'et': m_e}
        self.fungi_branch = fungi_branch  # dict: {'sel': sel_f, 'rf': m_r, 'et': m_e, 'lgb': m_l}
        self.tau_b        = tau_b
        self.tau_f        = tau_f
        self.bact_cols    = bact_cols
        self.fungi_cols   = fungi_cols

    def predict_proba(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Etapa 1: Predicción de dominio (0=Bacteria, 1=Fungi)
        X_all_mat = np.array(X_df, dtype=np.float32)
        pred_domains = self.domain_pipe.predict(X_all_mat)
        
        # Subsets por dominio
        if self.bact_cols is not None and isinstance(X, pd.DataFrame):
            X_bact = np.array(X_df[self.bact_cols], dtype=np.float32)
        else:
            X_bact = X_all_mat[:, :431] if X_all_mat.shape[1] > 431 else X_all_mat

        if self.fungi_cols is not None and isinstance(X, pd.DataFrame):
            X_fungi = np.array(X_df[self.fungi_cols], dtype=np.float32)
        else:
            X_fungi = X_all_mat

        p_cold = np.zeros(len(X_all_mat), dtype=np.float32)
        
        for i in range(len(X_all_mat)):
            if pred_domains[i] == 0:
                # Rama Bacteriana (431 features)
                xi = X_bact[[i]]
                xs = self.bact_branch['sel'].transform(xi)
                pl = self.bact_branch['lgb'].predict_proba(xs)[:, 1][0]
                pr = self.bact_branch['rf'].predict_proba(xs)[:, 1][0]
                pe = self.bact_branch['et'].predict_proba(xs)[:, 1][0]
                p_cold[i] = 0.50 * pl + 0.25 * pr + 0.25 * pe
            else:
                # Rama Fúngica (434 features)
                xi = X_fungi[[i]]
                xs = self.fungi_branch['sel'].transform(xi)
                pr = self.fungi_branch['rf'].predict_proba(xs)[:, 1][0]
                pe = self.fungi_branch['et'].predict_proba(xs)[:, 1][0]
                pl = self.fungi_branch['lgb'].predict_proba(xs)[:, 1][0]
                p_cold[i] = 0.40 * pr + 0.40 * pe + 0.20 * pl
                
        p_warm = 1.0 - p_cold
        return np.column_stack([p_cold, p_warm])

    def predict(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_mat = np.array(X_df, dtype=np.float32)
        pred_domains = self.domain_pipe.predict(X_mat)
        probs = self.predict_proba(X)[:, 0]
        
        preds = np.zeros(len(probs), dtype=int)
        for i in range(len(probs)):
            tau = self.tau_f if pred_domains[i] == 1 else self.tau_b
            # Thermal_Class: 0 = Cold, 1 = Warm
            preds[i] = 0 if probs[i] >= tau else 1
        return preds


# Backward compatibility alias
PsychroScanEnsemble = HierarchicalPsychroScan
