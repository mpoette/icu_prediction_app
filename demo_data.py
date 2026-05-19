"""
Mode démo — données ICU fictives.
Aucune connexion SQL ni modèle ML requis.
"""
from datetime import datetime, timedelta
import pandas as pd

_NOW = datetime.now().replace(minute=0, second=0, microsecond=0)

_SYS_COLS = [
    "sys_hémodynamique", "sys_respiratoire", "sys_neurologique",
    "sys_général", "sys_milieu_intérieur", "sys_hémostase", "sys_hépatique",
]

ALL_BEDS = [f"CH_{i:02d}" for i in range(1, 13)]
_ACTIVE_BEDS = [b for b in ALL_BEDS if b != "CH_11"]

ENCOUNTERS = {bed: 90000 + int(bed.split("_")[1]) for bed in _ACTIVE_BEDS}

# Scores sous-systèmes : valeurs brutes (0–1), la multiplication × score_global
# est effectuée par l'app (plot_radar_chart, df_corrected).
_P = {
    "CH_01": {
        "score": 0.82,
        "traj":  [0.62, 0.68, 0.72, 0.76, 0.80, 0.82],
        "sys":   [0.90, 0.75, 0.65, 0.80, 0.70, 0.60, 0.55],
        "vec": {"pam": 58, "pad": 38, "heart_rate": 118, "nad_dose_poids": 0.35,
                "spo2": 94, "fio2_corr": 60, "is_ventilated": 1,
                "is_conscious": 0, "is_sedated": 1, "is_not_alert": 1,
                "age": 68, "glyc_cap": 8.2, "creat": 185,
                "abs_dialyse": 1, "dialyse_hdi": 0, "dialyse_cvvhf": 0,
                "num_plq": 92, "tp": 52, "bili_tot": 28},
    },
    "CH_02": {
        "score": 0.55,
        "traj":  [0.63, 0.60, 0.59, 0.57, 0.56, 0.55],
        "sys":   [0.60, 0.72, 0.45, 0.50, 0.40, 0.55, 0.30],
        "vec": {"pam": 75, "pad": 52, "heart_rate": 95, "nad_dose_poids": 0.08,
                "spo2": 96, "fio2_corr": 45, "is_ventilated": 1,
                "is_conscious": 1, "is_sedated": 1, "is_not_alert": 0,
                "age": 72, "glyc_cap": 6.8, "creat": 95,
                "abs_dialyse": 1, "dialyse_hdi": 0, "dialyse_cvvhf": 0,
                "num_plq": 145, "tp": 78, "bili_tot": 15},
    },
    "CH_03": {
        "score": 0.28,
        "traj":  [0.44, 0.40, 0.36, 0.33, 0.30, 0.28],
        "sys":   [0.30, 0.25, 0.20, 0.28, 0.22, 0.35, 0.15],
        "vec": {"pam": 88, "pad": 62, "heart_rate": 78, "nad_dose_poids": 0.0,
                "spo2": 98, "fio2_corr": 21, "is_ventilated": 0,
                "is_conscious": 1, "is_sedated": 0, "is_not_alert": 0,
                "age": 58, "glyc_cap": 5.8, "creat": 82,
                "abs_dialyse": 1, "dialyse_hdi": 0, "dialyse_cvvhf": 0,
                "num_plq": 210, "tp": 92, "bili_tot": 12},
    },
    "CH_04": {
        "score": 0.78,
        "traj":  [0.58, 0.64, 0.68, 0.72, 0.76, 0.78],
        "sys":   [0.85, 0.80, 0.70, 0.75, 0.85, 0.65, 0.60],
        "vec": {"pam": 62, "pad": 42, "heart_rate": 105, "nad_dose_poids": 0.22,
                "spo2": 91, "fio2_corr": 70, "is_ventilated": 1,
                "is_conscious": 0, "is_sedated": 1, "is_not_alert": 1,
                "age": 74, "glyc_cap": 9.5, "creat": 245,
                "abs_dialyse": 0, "dialyse_hdi": 1, "dialyse_cvvhf": 0,
                "num_plq": 78, "tp": 45, "bili_tot": 35},
    },
    "CH_05": {
        "score": 0.48,
        "traj":  [0.42, 0.44, 0.45, 0.47, 0.48, 0.48],
        "sys":   [0.55, 0.40, 0.50, 0.45, 0.42, 0.48, 0.25],
        "vec": {"pam": 72, "pad": 50, "heart_rate": 88, "nad_dose_poids": 0.05,
                "spo2": 97, "fio2_corr": 35, "is_ventilated": 1,
                "is_conscious": 1, "is_sedated": 0, "is_not_alert": 0,
                "age": 65, "glyc_cap": 7.2, "creat": 128,
                "abs_dialyse": 1, "dialyse_hdi": 0, "dialyse_cvvhf": 0,
                "num_plq": 168, "tp": 72, "bili_tot": 18},
    },
    "CH_06": {
        "score": 0.22,
        "traj":  [0.38, 0.35, 0.31, 0.27, 0.24, 0.22],
        "sys":   [0.25, 0.20, 0.18, 0.22, 0.20, 0.28, 0.12],
        "vec": {"pam": 92, "pad": 65, "heart_rate": 72, "nad_dose_poids": 0.0,
                "spo2": 99, "fio2_corr": 21, "is_ventilated": 0,
                "is_conscious": 1, "is_sedated": 0, "is_not_alert": 0,
                "age": 45, "glyc_cap": 5.5, "creat": 68,
                "abs_dialyse": 1, "dialyse_hdi": 0, "dialyse_cvvhf": 0,
                "num_plq": 245, "tp": 98, "bili_tot": 8},
    },
    "CH_07": {
        "score": 0.91,
        "traj":  [0.75, 0.80, 0.84, 0.87, 0.89, 0.91],
        "sys":   [0.88, 0.95, 0.80, 0.85, 0.78, 0.72, 0.68],
        "vec": {"pam": 54, "pad": 35, "heart_rate": 125, "nad_dose_poids": 0.48,
                "spo2": 88, "fio2_corr": 80, "is_ventilated": 1,
                "is_conscious": 0, "is_sedated": 1, "is_not_alert": 1,
                "age": 78, "glyc_cap": 11.2, "creat": 320,
                "abs_dialyse": 0, "dialyse_hdi": 0, "dialyse_cvvhf": 1,
                "num_plq": 55, "tp": 38, "bili_tot": 52},
    },
    "CH_08": {
        "score": 0.61,
        "traj":  [0.75, 0.72, 0.68, 0.65, 0.63, 0.61],
        "sys":   [0.70, 0.55, 0.65, 0.60, 0.50, 0.58, 0.42],
        "vec": {"pam": 68, "pad": 46, "heart_rate": 92, "nad_dose_poids": 0.12,
                "spo2": 95, "fio2_corr": 40, "is_ventilated": 1,
                "is_conscious": 1, "is_sedated": 1, "is_not_alert": 0,
                "age": 62, "glyc_cap": 7.8, "creat": 142,
                "abs_dialyse": 1, "dialyse_hdi": 0, "dialyse_cvvhf": 0,
                "num_plq": 128, "tp": 68, "bili_tot": 20},
    },
    "CH_09": {
        "score": 0.35,
        "traj":  [0.50, 0.46, 0.43, 0.40, 0.37, 0.35],
        "sys":   [0.38, 0.30, 0.32, 0.35, 0.28, 0.40, 0.22],
        "vec": {"pam": 85, "pad": 60, "heart_rate": 82, "nad_dose_poids": 0.0,
                "spo2": 97, "fio2_corr": 28, "is_ventilated": 0,
                "is_conscious": 1, "is_sedated": 0, "is_not_alert": 0,
                "age": 55, "glyc_cap": 6.2, "creat": 105,
                "abs_dialyse": 1, "dialyse_hdi": 0, "dialyse_cvvhf": 0,
                "num_plq": 195, "tp": 85, "bili_tot": 14},
    },
    "CH_10": {
        "score": 0.75,
        "traj":  [0.52, 0.58, 0.64, 0.69, 0.72, 0.75],
        "sys":   [0.65, 0.50, 0.72, 0.70, 0.55, 0.68, 0.88],
        "vec": {"pam": 64, "pad": 44, "heart_rate": 98, "nad_dose_poids": 0.15,
                "spo2": 96, "fio2_corr": 38, "is_ventilated": 1,
                "is_conscious": 0, "is_sedated": 1, "is_not_alert": 1,
                "age": 52, "glyc_cap": 6.5, "creat": 178,
                "abs_dialyse": 1, "dialyse_hdi": 0, "dialyse_cvvhf": 0,
                "num_plq": 110, "tp": 42, "bili_tot": 145},
    },
    "CH_12": {
        "score": 0.52,
        "traj":  [0.48, 0.50, 0.51, 0.52, 0.52, 0.52],
        "sys":   [0.50, 0.42, 0.55, 0.48, 0.62, 0.45, 0.35],
        "vec": {"pam": 74, "pad": 52, "heart_rate": 86, "nad_dose_poids": 0.0,
                "spo2": 97, "fio2_corr": 30, "is_ventilated": 0,
                "is_conscious": 1, "is_sedated": 0, "is_not_alert": 0,
                "age": 70, "glyc_cap": 7.5, "creat": 225,
                "abs_dialyse": 0, "dialyse_hdi": 1, "dialyse_cvvhf": 0,
                "num_plq": 152, "tp": 75, "bili_tot": 22},
    },
}

CLINICAL_SUMMARY = {"n_ventile": 7, "n_nad": 5, "n_dialyse": 3, "n_febrile": 4}


def get_all_service_beds():
    return list(ALL_BEDS)


def get_active_encounters():
    return dict(ENCOUNTERS)


def _make_df(bed, data):
    rows = []
    for i, score in enumerate(data["traj"]):
        t = _NOW - timedelta(hours=5 - i)
        row = {"date_calcul": t, "lit": bed,
               "encounterId": ENCOUNTERS[bed], "score_sévérité": score}
        for col, val in zip(_SYS_COLS, data["sys"]):
            row[col] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    df["date_calcul"] = pd.to_datetime(df["date_calcul"])
    return df.sort_values("date_calcul").reset_index(drop=True)


def get_patient_df(bed):
    if bed not in _P:
        return pd.DataFrame()
    return _make_df(bed, _P[bed])


def get_global_df():
    frames = [_make_df(bed, data) for bed, data in _P.items()]
    return pd.concat(frames, ignore_index=True)


def get_feature_vector(bed):
    return dict(_P.get(bed, {}).get("vec", {}))
