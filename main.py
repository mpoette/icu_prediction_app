# /// script
# dependencies = [
#   "pandas>=2.0.0",
#   "numpy>=1.24.0",
#   "pyarrow>=12.0.0",       
#   "scikit-learn>=1.2.0",   
#   "joblib>=1.2.0",         
#   "torch>=2.0.0",          
#   "matplotlib>=3.7.0",    
#   "SQLAlchemy>=2.0.0",     
#   "pyodbc>=4.0.39",     
#   "pymssql==2.3.12",   
#   "streamlit>=1.20.0",
#   "tqdm",
# ]
# ///

import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import extract
from transform import preprocessing
from transform import predict
import load
import plot
import utils
import os
import config as cfg

MAX_WORKERS = min((os.cpu_count() or 2) * 2, 8)

_UTC_OFFSET = timedelta(seconds=(datetime.now() - datetime.utcnow()).total_seconds())

def utc_to_local(utc_ts):
    return pd.Timestamp(utc_ts) + _UTC_OFFSET


def process_history(lit):
    try:
        df_raw, official_label = extract.load_data_from_sql(target_bed=lit)
        if df_raw.empty:
            print(f" [INFO] Aucune donnée pour le lit {lit}.")
            return

        history_results = []
        local_max_time = utc_to_local(pd.to_datetime(df_raw[cfg.COL_DATE_MESURE]).max())
        print(f"\n{'='*60}\n HISTORIQUE DU PATIENT LIT {official_label}\n{'='*60}")
        print(f" ┌── [PRÉTRAITEMENT & INFÉRENCE LSTM]")

        for h in range(6):
            df_clean = preprocessing.prepare_data(df_raw, hour_offset=h)
            if not df_clean.empty:
                pids, scores, sys_scores = predict.run_algo(df_clean)
                if len(scores) > 0:
                    timestamp = (local_max_time - timedelta(hours=h)).strftime('%Y-%m-%d %H:%M:%S')
                    sub = {k: round(float(v[0]),4) for k, v in sys_scores.items()}
                    history_results.append({
                        'date_calcul':    timestamp,
                        'encounterId':    df_raw[cfg.ID_COL].iloc[0],
                        'score_sévérité': round(float(scores[0]), 4),
                        **sub
                    })

        if history_results:
            load.save_patient_history(official_label, history_results)
            latest = history_results[0]
            sub = {k: v for k, v in latest.items() if str(k).startswith('sys_')}
            load.save_results(official_label, df_raw[cfg.ID_COL].iloc[0],
                              latest['score_sévérité'],
                              timestamp=latest['date_calcul'],
                              sub_scores=sub)
            plot.plot_patient_trajectory(official_label)

    except Exception as e:
        print(f" [ERREUR HISTORIQUE] {e}")


def process_bed_simple(lit, reference_utc, reference_local, hour_offset=0):
    print(f" ├── [ANALYSE] Lit {lit}" + (f" ({reference_local})" if hour_offset else ""))
    try:
        df_raw, label = extract.load_data_from_sql(lit, reference_utc=reference_utc)
        if not df_raw.empty:
            df_raw = df_raw[pd.to_datetime(df_raw[cfg.COL_DATE_MESURE]) <= reference_utc].copy()
            if df_raw.empty:
                print(f" │   └─ ✕ {label} : aucune mesure avant {reference_local}")
                return lit, None
            df_clean = preprocessing.prepare_data(df_raw, hour_offset=0)
            if not df_clean.empty:
                pids, scores, sys_scores = predict.run_algo(df_clean)
                if len(scores) > 0:
                    sub = {k: round(float(v[0]),4) for k, v in sys_scores.items()}
                    load.save_results(label, pids[0], scores[0], timestamp=reference_local, sub_scores=sub)
                    print(f" │   └─ OK  {label} → {scores[0]:.4f}  [{reference_local}]")
                    return label, float(scores[0])
    except Exception as e:
        print(f" │   └─ ERREUR {lit} : {e}")
    return lit, None


def process_single_hour(lit, hour_offset):
    try:
        reference_utc = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hour_offset)
        reference_local = utc_to_local(reference_utc).strftime('%Y-%m-%d %H:%M:%S')

        df_raw, official_label = extract.load_data_from_sql(target_bed=lit, reference_utc=reference_utc)
        if df_raw.empty: return

        df_clean = preprocessing.prepare_data(df_raw, hour_offset=0)
        if df_clean.empty: return

        pids, scores, sys_scores = predict.run_algo(df_clean)
        if len(scores) == 0: return

        sub = {k: round(float(v[0]), 4) for k, v in sys_scores.items()}
        
        # PERSISTANCE DANS L'HISTORIQUE DU LIT
        safe_label = str(official_label).replace('/', '_')
        path_lit = cfg.OUTPUT_DIR / f"historique_lit_{safe_label}.csv"
        
        nouveau_point = {
            'date_calcul': reference_local,
            'encounterId': df_raw[cfg.ID_COL].iloc[0],
            'score_sévérité': round(float(scores[0]), 4),
            **sub
        }

        if path_lit.exists():
            df_hist = pd.read_csv(path_lit, sep=';')
            # Suppression d'un éventuel doublon à la même heure
            df_hist = df_hist[df_hist['date_calcul'] != reference_local]
            df_hist = pd.concat([df_hist, pd.DataFrame([nouveau_point])], ignore_index=True)
        else:
            df_hist = pd.DataFrame([nouveau_point])

        # Sauvegarde via la fonction existante
        load.save_patient_history(official_label, df_hist.to_dict('records'))
        
        # Signal pour Streamlit
        print(f"SCORE_RESULT {scores[0]:.4f} | {official_label} | H-{hour_offset} | {reference_local}")

    except Exception as e:
        print(f" [ERREUR PONCTUEL] {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lit", type=str, nargs="?", default=None)
    parser.add_argument("--w", type=int, default=MAX_WORKERS)
    parser.add_argument("--h", type=int, default=None, dest="hour")
    
    args = parser.parse_args()

    if args.lit and args.hour is not None:
        process_single_hour(args.lit, args.hour)

    elif args.lit:
        process_history(args.lit)

    else:
        hour_offset = args.hour if args.hour is not None else 0
        label_h = f" à H-{hour_offset}" if hour_offset else " (temps réel)"
        print(f"\n{'='*60}")
        print(f"  MODE SERVICE : BALAYAGE PARALLÈLE (workers={args.w}){label_h}")
        print(f"{'='*60}")

        engine = utils.get_db_engine(cfg.SQL_SERVER, cfg.SQL_DATABASE)

        reference_utc   = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hour_offset)
        reference_local = utc_to_local(reference_utc).strftime('%Y-%m-%d %H:%M:%S')
        print(f" Instant de référence : {reference_local}\n")

        patients = extract.get_all_patients_service(engine, reference_utc=reference_utc)
        if not patients:
            print(" [INFO] Aucun patient trouvé dans le service.")
            return

        print(f" {len(patients)} lits à analyser\n")
        start = datetime.now()

        with ThreadPoolExecutor(max_workers=args.w) as executor:
            futures = {
                executor.submit(process_bed_simple, p['bedLabel'], reference_utc, reference_local, hour_offset): p['bedLabel']
                for p in patients
            }
            ok_count, err_count = 0, 0
            for future in as_completed(futures):
                label, score = future.result()
                if score is not None:
                    ok_count += 1
                else:
                    err_count += 1

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n{'='*60}")
        print(f"  TERMINÉ en {elapsed:.1f}s — {ok_count} OK / {err_count} erreur(s)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()