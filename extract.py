import pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import config as cfg
import utils


def _reference_utc(hour_offset: int) -> datetime:
    """Calcule l'instant de reference en UTC depuis l'heure locale du systeme."""
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    return now_utc - __import__('datetime').timedelta(hours=hour_offset)


def _fmt_utc(dt: datetime) -> str:
    """Formate un datetime UTC pour injection SQL (YYYY-MM-DD HH:MM:SS)."""
    return dt.strftime("'%Y-%m-%d %H:%M:%S'")


def get_patient_in_bed(engine, target_bed=None, reference_utc: datetime = None):
    """Identifie le patient present dans le lit a l'instant de reference UTC."""
    clean_lit = str(target_bed).lower().strip()

    if reference_utc is None:
        time_filter = "utcOutTime IS NULL"
        time_in     = "GETUTCDATE()"
    else:
        ref = _fmt_utc(reference_utc)
        time_filter = f"(utcOutTime IS NULL OR utcOutTime > {ref})"
        time_in     = ref

    query = f"""
    SELECT encounterId, utcInTime, bedLabel 
    FROM PtBedStay 
    WHERE clinicalUnitId = {cfg.ID_SERVICE_REA} 
      AND {time_filter}
      AND utcInTime <= {time_in}
      AND bedId != 1
      AND (LOWER(bedLabel) LIKE '%{clean_lit}%' OR LOWER(bedLabel) LIKE '%0{clean_lit}%')
    """
    try:
        df = pd.read_sql_query(query, engine)
        if df.empty: return None
        df['len'] = df['bedLabel'].str.len()
        patient = df.sort_values('len').iloc[0].to_dict()

        q_age = f"SELECT age FROM D_Encounter WHERE encounterId = {patient['encounterId']}"
        df_age = pd.read_sql_query(q_age, engine)
        patient['age'] = float(df_age.iloc[0]['age']) if not df_age.empty else 60.0
        return patient
    except Exception as e:
        print(f" [ERREUR LIT] {e}")
        return None


def get_all_patients_service(engine, reference_utc: datetime = None):
    """Extrait tous les lits occupes du service a l'instant de reference UTC."""
    if reference_utc is None:
        time_filter = "utcOutTime IS NULL"
        time_in     = "GETUTCDATE()"
    else:
        ref = _fmt_utc(reference_utc)
        time_filter = f"(utcOutTime IS NULL OR utcOutTime > {ref})"
        time_in     = ref

    query = (
        f"SELECT encounterId, bedLabel, utcInTime FROM PtBedStay "
        f"WHERE clinicalUnitId = {cfg.ID_SERVICE_REA} "
        f"AND {time_filter} "
        f"AND utcInTime <= {time_in} "
        f"AND bedLabel IS NOT NULL AND bedId != 1"
    )
    try:
        df = pd.read_sql_query(query, engine)
        if df.empty: return []
        df['bed_num'] = df['bedLabel'].str.extract(r'(\d+)').astype(float)
        df = df.sort_values(by=['bed_num', 'bedLabel'])
        return df.drop(columns=['bed_num']).to_dict('records')
    except Exception as e:
        print(f" [ERREUR LISTE] {e}")
        return []


def load_data_from_sql(target_bed=None, reference_utc: datetime = None):
    """Point d'entree principal utilisant le thesaurus."""
    engine = utils.get_db_engine(cfg.SQL_SERVER, cfg.SQL_DATABASE)
    patient = get_patient_in_bed(engine, target_bed, reference_utc=reference_utc)
    if not patient: raise ValueError(f"Lit {target_bed} introuvable.")

    print(f" │   ├─ Patient identifié (ID: {patient['encounterId']} | Age: {patient['age']})")

    pid = patient['encounterId']
    thesaurus = utils.load_thesaurus('thesaurus.json')

    features = [(fid, fcfg) for fid, fcfg in thesaurus["features"].items()
                if fcfg.get("type") != "static"]

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(utils.extract_feature, engine, pid, fcfg): fid
                   for fid, fcfg in features}
        for future in as_completed(futures):
            try:
                df_feat = future.result()
                if not df_feat.empty:
                    results.append(df_feat)
            except Exception as e:
                print(f" │   [WARN extract] {futures[future]} : {e}")

    if not results:
        print(f" │   └─ ✕ Aucune mesure clinique exploitable.")
        return pd.DataFrame(), patient['bedLabel']

    df_final = pd.concat(results, axis=1).sort_index().reset_index()
    df_final = df_final.rename(columns={'utcChartTime': cfg.COL_DATE_MESURE})
    df_final[cfg.ID_COL] = str(pid)
    df_final['official_bed_label'] = patient['bedLabel']
    df_final['age'] = patient['age']
    df_final[cfg.COL_DATE_ADMISSION] = pd.to_datetime(patient['utcInTime'])
    df_final[cfg.TIME_COL] = (df_final[cfg.COL_DATE_MESURE] - df_final[cfg.COL_DATE_ADMISSION]).dt.total_seconds() / 3600

    return df_final, patient['bedLabel']