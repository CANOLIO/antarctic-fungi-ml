import numpy as np

class PsychroScanEnsemble:
    """
    Ensemble biofísico multimodelo (LightGBM + Random Forest + ExtraTrees)
    con Feature Denoising por Mutual Information.
    Garantiza compatibilidad 100% con los scripts downstream:
      predict_proba(X)[:, 0] -> P(Cold)
      predict_proba(X)[:, 1] -> P(Warm)
    """
    def __init__(self, m_lgb, m_rf, m_et, selector=None):
        self.m_lgb = m_lgb
        self.m_rf = m_rf
        self.m_et = m_et
        self.selector = selector

    def predict_proba(self, X):
        X_mat = np.array(X, dtype=np.float32)
        if self.selector is not None:
            X_mat = self.selector.transform(X_mat)
        p_lgb = self.m_lgb.predict_proba(X_mat)[:, 1]
        p_rf  = self.m_rf.predict_proba(X_mat)[:, 1]
        p_et  = self.m_et.predict_proba(X_mat)[:, 1]
        p_cold = 0.50 * p_lgb + 0.25 * p_rf + 0.25 * p_et
        p_warm = 1.0 - p_cold
        return np.column_stack([p_cold, p_warm])

    def predict(self, X, threshold=0.25):
        probs = self.predict_proba(X)[:, 0]
        return np.where(probs >= threshold, 0, 1)
