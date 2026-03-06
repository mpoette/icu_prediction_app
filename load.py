import pandas as pd
import os
import threading
from pathlib import Path
import config as cfg


file_lock = threading.Lock()


COLUMNS = [
    'date_calcul', 'lit', 'encounterId', 'score_sévérité',
    'sys_hémodynamique', 'sys_respiratoire', 'sys_neurologique',
    'sys_infectieux', 'sys_général',
    'sys_milieu_intérieur', 'sys_hémostase', 'sys_hépatique'
]

def save_results(bed_label, encounter_id, score, timestamp, sub_scores=None):
    """Ajoute une ligne au registre global avec une structure strictement fixe."""
    filepath = cfg.OUTPUT_DIR / "historique_scores_rea.csv"
    

    row_data = {col: None for col in COLUMNS}
    
    # Remplissage des données de base
    row_data.update({
        'date_calcul':    timestamp,
        'lit':            bed_label,
        'encounterId':    encounter_id,
        'score_sévérité': round(float(score), 4)
    })
    
    # Remplissage des sous-scores
    if sub_scores:
        for k, v in sub_scores.items():
            if k in COLUMNS:
                row_data[k] = round(float(v), 4)
            
    df_new = pd.DataFrame([row_data])
    
    with file_lock:
        # forcer ordre des colonnes avant l'écriture
        df_new = df_new[COLUMNS]
        if filepath.exists():
            df_new.to_csv(filepath, sep=';', mode='a', header=False, index=False, encoding='utf-8')
        else:
            df_new.to_csv(filepath, sep=';', mode='w', header=True, index=False, encoding='utf-8')

def save_patient_history(bed_label, history_data):
    """Génère l'historique 6h pour un lit (utilisé pour les graphiques)."""
    if not history_data: return
    
    safe_label = str(bed_label).replace('/', '_')
    filepath = cfg.OUTPUT_DIR / f"historique_lit_{safe_label}.csv"
    
    df_hist = pd.DataFrame(history_data)
    for col in COLUMNS:
        if col not in df_hist.columns and col != 'Δ': 
            df_hist[col] = None
            
    df_hist = df_hist.sort_values(by='date_calcul', ascending=True)
    
    with file_lock:
        df_hist.to_csv(filepath, sep=';', index=False, encoding='utf-8')