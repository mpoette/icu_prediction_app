import joblib
import numpy as np
import config as cfg
import torch
import os
from pathlib import Path
from config import MODEL_WEIGHTS_PATH, MODEL_PKL_PATH, MODELS_DIR
from models.lstmTimeModified import load_model_from_checkpoint, predict_proba


def _load_system_models():
    """
    Charge tous les modèles par label depuis MODELS_LABELS_DIR.
    Format attendu (sauvé par le notebook) :
        {model_name}_modele_complet.pkl  →  {'scaler', 'model_path_weights', 'metadata'}
        {model_name}_best_weights.pt     →  poids PyTorch
    Retourne un dict {label: {'model', 'T', 'scaler', 'features'}}.
    """
    labels_dir = cfg.MODELS_LABELS_DIR
    if not labels_dir.exists():
        return {}

    sys_models = {}
    for pkl_path in sorted(labels_dir.glob('*_modele_complet.pkl')):
        try:
            pack  = joblib.load(pkl_path)
            meta  = pack.get('metadata', {})
            label = meta.get('label', pkl_path.stem.replace('_modele_complet', ''))

            pt_path = Path(pack.get('model_path_weights', ''))
            if not pt_path.exists():
                pt_path = labels_dir / f"{label}_best_weights.pt"
            if not pt_path.exists():
                print(f" [WARN] Poids introuvables pour label '{label}' ({pt_path})")
                continue

            model, _, T = load_model_from_checkpoint(str(pt_path))
            T = meta.get('temperature', T)

            sys_models[label] = {
                'model':    model,
                'T':        float(T),
                'scaler':   pack['scaler'],
                'features': list(meta.get('features', pack['scaler'].feature_names_in_)),
            }
            print(f" [INFO] Modèle label '{label}' chargé (AUC={meta.get('auc_test', '?'):.4f})")
        except Exception as e:
            print(f" [WARN] Impossible de charger {pkl_path.name} : {e}")

    return sys_models


def run_algo(df_clean):
    """
    Inférence stricte basée uniquement sur l'imputation générée en amont.
    """
    if not cfg.MODEL_PKL_PATH.exists():
        raise FileNotFoundError(f"Transformateur introuvable : {cfg.MODEL_PKL_PATH}")

    df_sorted   = df_clean.sort_values(by=[cfg.ID_COL, 'heure_calibree'])
    ids_uniques = df_sorted[cfg.ID_COL].unique()

    sys_scores_matrix = {}
    sys_models = _load_system_models()

    if sys_models:
        for label, sm in sys_models.items():
            key = f'sys_{label}'
            try:
                features_list = sm['features']

                missing = [c for c in features_list if c not in df_sorted.columns]
                if missing:
                    raise ValueError(f"Colonnes manquantes (échec imputation) : {missing}")

                df_sys = df_sorted[features_list].copy()

                X_sys_scaled = sm['scaler'].transform(df_sys)
                X_sys_3d     = X_sys_scaled.reshape(len(ids_uniques), cfg.WINDOW_SIZE, len(features_list))
                probas_sys   = predict_proba(sm['model'], X_sys_3d, T=sm['T'])
                sys_score    = probas_sys[:, 1] if (probas_sys.ndim > 1 and probas_sys.shape[1] > 1) else probas_sys.flatten()
                
                sys_scores_matrix[key] = sys_score

            except Exception as e:
                print(f" [WARN] Inférence label '{label}' échouée : {e}")
                sys_scores_matrix[key] = np.full(len(ids_uniques), np.nan)
    else:
        print(" [ERREUR] Aucun sous-modèle trouvé.")

    # ── Calcul du Score Global (Modèle LSTM complet) ──────────────────
    if 'sys_global' in sys_scores_matrix:
        scores = sys_scores_matrix.pop('sys_global')
        type_score = "Global (LSTM)"
    else:
        scores = np.full(len(ids_uniques), np.nan)
        type_score = "INDISPONIBLE"

    # ── Affichage Console & Debug ────────────────────────────────────────────
    features_debug = cfg.FEATURES_LIST
    
    for i, patient_id in enumerate(ids_uniques):
        df_patient = df_sorted[df_sorted[cfg.ID_COL] == patient_id]
        row_h0 = df_patient[df_patient['heure_calibree'] == 0]
        
        if not row_h0.empty:
            print(f" │   │   └─ Score Sévérité {type_score} : {scores[i]:.4f}")
            print(f" │   │   └─ [DEBUG] Vecteur clinique injecté (H=0) :")
            
            # Extraction des valeurs pour chaque variable du thésaurus présente dans le df
            debug_vals = [f"{col}: {row_h0[col].values[0]:.2f}" for col in features_debug if col in row_h0.columns]
            
            # Affichage sur plusieurs colonnes (par paquets de 4)
            for j in range(0, len(debug_vals), 4):
                print(f" │   │      " + " │ ".join(debug_vals[j:j+4]))
        else:
            print(f" │   │   └─ Score Sévérité {type_score} : {scores[i]:.4f} (Données H=0 indisponibles)")

    return ids_uniques, scores, sys_scores_matrix