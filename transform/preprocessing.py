import pandas as pd
import numpy as np
import config as cfg
import utils

def prepare_data(df, hour_offset=0):
    thesaurus = utils.load_thesaurus('thesaurus.json')
    
    if not df.empty and cfg.COL_DATE_MESURE in df.columns:
        real_max_time = pd.to_datetime(df[cfg.COL_DATE_MESURE]).max()
        simulated_time = real_max_time - pd.Timedelta(hours=hour_offset)
        time_str = simulated_time.strftime('%H:%M')
    else:
        time_str = "XX:XX"
        
    df['heure_entiere'] = np.floor(df[cfg.TIME_COL])
    max_time = df.groupby(cfg.ID_COL)['heure_entiere'].transform('max') - hour_offset
    df['heure_calibree'] = df['heure_entiere'] - max_time
    
    # Historique jusqu'à l'heure cible
    df_history = df[df['heure_calibree'] <= 0].copy()
    
    if df_history.empty:
        print(f" │   ├─ {time_str} (H-{hour_offset:<2}) ── ✕ Blocage : Historique vide")
        return pd.DataFrame()

    # Agrégation sur tout l'historique
    df_agg = utils.apply_generic_aggregation(df_history, thesaurus, cfg.ID_COL, 'heure_calibree')
    
    # Réindexation depuis le début de l'hospitalisation jusqu'à 0
    ids = df_agg.index.get_level_values(cfg.ID_COL).unique()
    min_h = int(df_history['heure_calibree'].min())
    min_h = min(min_h, -cfg.WINDOW_SIZE + 1) # Assure d'avoir au moins 24h
    
    idx_full = pd.MultiIndex.from_product(
        [ids, range(min_h, 1)], 
        names=[cfg.ID_COL, 'heure_calibree']
    )
    df_full = df_agg.reindex(idx_full)
    
    # Imputation 
    df_full = utils.apply_generic_imputation(df_full, thesaurus)
    
    # Découpage de la fenêtre de 24h stricte pour le LSTM
    df_window_final = df_full.loc[(slice(None), slice(-cfg.WINDOW_SIZE + 1, 0)), :].copy()

    # --- GESTION DE LA DIALYSE ---
    for _col, _def in [('dialyse_hdi', 0), ('dialyse_cvvhf', 0), ('abs_dialyse', 1)]:
        if _col not in df_window_final.columns:
            df_window_final[_col] = _def
            
    mask_absent = (df_window_final['dialyse_hdi'] == 0) & (df_window_final['dialyse_cvvhf'] == 0)
    df_window_final['abs_dialyse'] = mask_absent.astype(int)
    
    print(f" │   ├─ {time_str} (H-{hour_offset}) ")
    
    return df_window_final.reset_index()