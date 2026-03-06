import os
import json

# ============================================================
# PATHS ET MODELE
# ============================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#  chemins relatifs
THESAURUS_PATH = BASE_DIR / "thesaurus.json"
MODELS_DIR = BASE_DIR / "models"

MODEL_WEIGHTS_PATH  = MODELS_DIR / "best_lstm_weights.pt"
MODEL_PKL_PATH      = MODELS_DIR / "modele_lstm_complet.pkl"

# Dossier des modèles par label
MODELS_LABELS_DIR = MODELS_DIR / "models_labels"
MODELS_LABELS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG SQL
# ============================================================
SQL_SERVER     = 'SVM-ICCA-DAR-P'
SQL_DATABASE   = 'CISReportingActiveDB0'
ID_SERVICE_REA = 3
TARGET_BED     = '22'

# ============================================================
# NOMS DES COLONNES TECHNIQUES (fixes, indépendants du thésaurus)
# ============================================================
COL_DATE_MESURE    = 'utcChartTime'
COL_DATE_ADMISSION = 'utcInTime'
COL_VALEUR_SOURCE  = 'valueNumber'
COL_PROP_NAME      = 'dictionaryPropName'
COL_LAB_LABEL      = 'labTestLabel'
COL_INV            = 'pam_invasive'
COL_NON_INV        = 'pam_non_invasive'
FINAL_FEATURE      = 'pam'

ID_COL      = 'encounterId'
TIME_COL    = 'delta_hour'
WINDOW_SIZE = 24



def _load_thesaurus(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_cols_from_thesaurus(thesaurus: dict) -> dict:
    """
    Construit {feature_id: nom_colonne} depuis le thésaurus.
    """
    cols = {}
    for feat_id, feat_cfg in thesaurus['features'].items():
        if 'column' in feat_cfg:
            cols[feat_id] = feat_cfg['column']
        elif 'columns' in feat_cfg:
            cols[feat_id] = feat_cfg['columns']
    return cols


def _build_feature_list(thesaurus: dict) -> list:
    """
    Retourne la liste ORDONNÉE de toutes les colonnes numériques
    produites par le preprocessing (dans l'ordre du thésaurus).
    Doit correspondre à scaler.feature_names_in_.
    Les features de type 'static' sont ignorées.
    """
    cols = []
    for feat_id, feat_cfg in thesaurus['features'].items():
        if feat_cfg.get('type') in ('static', 'source_only', 'derived'):
            continue
        if 'column' in feat_cfg:
            cols.append(feat_cfg['column'])
        elif 'columns' in feat_cfg:
            cols.extend(feat_cfg['columns'])
    return cols

def _build_physio_map(thesaurus: dict) -> dict:
    """
    Construit la cartographie { 'Système': ['col1', 'col2'] } 
    en lisant la clé 'categories' du thésaurus.
    """
    physio_map = {}
    for feat_id, feat_cfg in thesaurus.get('features', {}).items():
        # Extraction des noms de colonnes
        cols = []
        if 'column' in feat_cfg:
            cols.append(feat_cfg['column'])
        elif 'columns' in feat_cfg:
            cols.extend(feat_cfg['columns'])
            
        # Assignation aux labels
        cats = feat_cfg.get('label', []) 
        for cat in cats:
            if cat not in physio_map:
                physio_map[cat] = []
            physio_map[cat].extend(cols)
            
    return physio_map

try:
    _thesaurus = _load_thesaurus(THESAURUS_PATH)


    COLS_FROM_THESAURUS = _build_cols_from_thesaurus(_thesaurus)
    # Liste ordonnée des colonnes produites
    FEATURES_LIST = _build_feature_list(_thesaurus)
    PHYSIO_MAP = _build_physio_map(_thesaurus)

except FileNotFoundError:
    print(f" [WARN] thesaurus.json introuvable ({THESAURUS_PATH}).")
    _thesaurus, COLS_FROM_THESAURUS, FEATURES_LIST, PHYSIO_MAP = {}, {}, [], {}
except FileNotFoundError:
    print(f" [WARN] thesaurus.json introuvable ({THESAURUS_PATH}). COLS_FROM_THESAURUS vide.")
    _thesaurus        = {}
    COLS_FROM_THESAURUS = {}
    FEATURES_LIST     = []
except Exception as e:
    print(f" [ERREUR config_u] Chargement thésaurus : {e}")
    _thesaurus        = {}
    COLS_FROM_THESAURUS = {}
    FEATURES_LIST     = []
    PHYSIO_MAP = {}