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

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import subprocess, sys, os
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="SEVERITE REA", page_icon="", layout="wide")

st.markdown("""
<style>
  .bed-score { font-size:1.7rem; font-weight:700; font-family:monospace; line-height:1.1; }
  .ok   { color:#2e7d32 } .warn { color:#f57c00 } .crit { color:#c62828 } .none { color:#888 }
  .badge { 
      display:inline-block; 
      padding:4px 12px; 
      border-radius:20px;
      font-size:0.85rem; 
      font-weight:700; 
      font-family:monospace; 
      letter-spacing: 0.5px;
  }
  .badge-ok   { background:#e8f5e9; color:#2e7d32 }
  .badge-warn { background:#fff3e0; color:#f57c00 }
  .badge-crit { background:#ffebee; color:#c62828 }
  .badge-none { background:#f0f0f0; color:#888 }
  .badge-vide { background:#f0f0f0; color:#aaa; font-style:italic }
  div[data-testid="stButton"] button {
      border-radius: 4px;
      font-weight: 500;
      font-size: 0.9rem;
      border: 1px solid #cbd5e1;
      background-color: white;
      color: #334155;
      transition: all 0.15s ease;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      box-shadow: 0 4px 6px rgba(0,0,0,0.08);
  }
    .sys-card {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px;
      text-align: center;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
  }
  .sys-label { font-size: 0.65rem; color: #64748b; text-transform: uppercase; font-weight: 700; margin-bottom: 4px; }
  .sys-value { font-size: 1rem; font-weight: 800; font-family: monospace; }
            /* Style pour les onglets segmentés */
    div[data-testid="stSegmentedControl"] {
        gap: 8px;
    }
    div[data-testid="stSegmentedControl"] button {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #64748b;
        transition: all 0.2s;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background-color: #eff6ff !important;
        border-color: #3b82f6 !important;
        color: #1d4ed8 !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    /* Carte de détail patient */
    .detail-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
</style>""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)
CSV  = OUT / "historique_scores_rea.csv"

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("selected_bed", None), ("pending_run", None), ("run_logs", ""),
             ("last_execution_log", ""),
             ("run_ok", None), ("pending_hour_run", None), ("hour_score_result", None),
             ("hour_score_label", ""), ("hour_score_vector", {}), ("main_score_vector", {}), ("cache_version", 0), ("last_scan_hour", 0), ("last_scan_ref", ""), ("last_scan_utc", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_global(_version=0):
    if not CSV.exists():
        cols = ['date_calcul', 'lit', 'encounterId', 'score_sévérité',
                'sys_hémodynamique', 'sys_respiratoire', 'sys_neurologique', 
                'sys_infectieux', 'sys_général']
        return pd.DataFrame(columns=cols)
    
    try:
        df = pd.read_csv(CSV, sep=";", on_bad_lines='skip', low_memory=False)
        df["date_calcul"] = pd.to_datetime(df["date_calcul"])
        return df
    except Exception as e:
        st.error(f"Erreur de lecture du fichier global : {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_patient(bed, _version=0):
    safe_label = str(bed).replace('/', '_')
    fp = OUT / f"historique_lit_{safe_label}.csv"
    if not fp.exists(): return pd.DataFrame()
    
    df = pd.read_csv(fp, sep=";")
    df["date_calcul"] = pd.to_datetime(df["date_calcul"])
    return df.sort_values("date_calcul")

@st.cache_data(ttl=60)
def get_active_encounters(reference_utc=None, _version=0):
    """Recupere les patients presents dans le service a l'instant reference_utc (ou maintenant si None)."""
    try:
        sys.path.insert(0, str(BASE))
        import utils, config as cfg
        engine = utils.get_db_engine(cfg.SQL_SERVER, cfg.SQL_DATABASE)
        import extract
        patients = extract.get_all_patients_service(engine, reference_utc=reference_utc)
        return {str(p["bedLabel"]): p["encounterId"] for p in patients}
    except Exception:
        return {}

@st.cache_data(ttl=60)
def get_all_service_beds():
    """Tous les lits physiques du service, qu'ils soient occupés ou vides."""
    try:
        sys.path.insert(0, str(BASE))
        import utils, config as cfg
        engine = utils.get_db_engine(cfg.SQL_SERVER, cfg.SQL_DATABASE)
        q = f"""
            SELECT DISTINCT bedLabel FROM PtBedStay
            WHERE clinicalUnitId = 3
            AND bedLabel IS NOT NULL
            AND bedId != 1  
            AND bedLabel NOT LIKE '%P'
            ORDER BY bedLabel
        """
        df = pd.read_sql_query(q, engine)

        beds = df["bedLabel"].dropna().unique().tolist()
        return sorted(beds, key=lambda b: int(''.join(filter(str.isdigit, str(b))) or 0))
    except Exception:
        return []

def run_pipeline(bed="", mode="global", hour_offset=0):
    cmd = [sys.executable, str(BASE/"main.py")] + ([str(bed)] if mode=="history" else [])
    if hour_offset:
        cmd += ["--h", str(hour_offset)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=300, env=env, encoding="utf-8", errors="replace")
        return r.returncode == 0, r.stdout + r.stderr
    except Exception as e:
        return False, str(e)

def run_pipeline_at_hour(bed, hour_offset: int):
    """Lance une prédiction ponctuelle via main.py --h N."""
    cmd = [sys.executable, str(BASE / "main.py"), str(bed), "--h", str(hour_offset)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=120, env=env, encoding="utf-8", errors="replace")
        output = r.stdout + r.stderr
        score, parsed_h, parsed_time, vector = None, hour_offset, "", {}
        lines = output.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("SCORE_RESULT"):
                try:
                    parts = [p.strip() for p in line.split("|")]
                    score = float(parts[0].split()[1])
                    for p in parts:
                        if p.startswith("H-"): parsed_h = int(p.replace("H-",""))
                    if len(parts) >= 4: parsed_time = parts[-1].strip()
                except Exception:
                    pass
            if "[DEBUG]" in line:
                for debug_line in lines[i+1:]:
                    if "│" not in debug_line and ":" not in debug_line:
                        break
                    for pair in debug_line.split("│"):
                        pair = pair.strip()
                        if ":" in pair:
                            try:
                                k, v = pair.split(":", 1)
                                vector[k.strip()] = float(v.strip())
                            except Exception:
                                pass
        return r.returncode == 0, output, score, parsed_h, parsed_time, vector
    except Exception as e:
        return False, str(e), None, hour_offset, ""


def badge(score):
    if score is None: return "–", "none", "N/A"
    if score >= 0.75: return f"{score:.2f}", "crit", "CRITIQUE"
    if score >= 0.50: return f"{score:.2f}", "warn", "VIGILANCE"
    return f"{score:.2f}", "ok", "STABLE"

def get_score_colors(score):
    """Retourne un tuple (background, texte/bordure) selon le score."""
    if score is None: return "#f0f0f0", "#888"
    if score >= 0.75: return "#ffebee", "#c62828"  # Rouge (Critique)
    if score >= 0.50: return "#fff3e0", "#f57c00"  # Orange (Vigilance)
    return "#e8f5e9", "#2e7d32"

#  Plot 
def plot_trajectory(df, bed, col="score_sévérité"):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axhspan(.75, 1.05, color="#ffebee", alpha=.5)
    ax.axhspan(.50, .75,  color="#fff8e1", alpha=.5)
    
    dates = df["date_calcul"].values
    scores = df[col].values
    
    for i in range(len(scores)-1):
        if pd.isna(scores[i]) or pd.isna(scores[i+1]): continue
        c = "#c62828" if scores[i]>=.75 else "#f57c00" if scores[i]>=.50 else "#2e7d32"
        ax.plot(dates[i:i+2], scores[i:i+2], color=c, lw=2.5)
        
    for d, s in zip(dates, scores):
        if pd.isna(s): continue
        c = "#c62828" if s>=.75 else "#f57c00" if s>=.50 else "#2e7d32"
        ax.scatter(d, s, color=c, s=50, zorder=5, edgecolors="white", lw=1.5)
    
    ax.set_ylim(-0.05, 1.05)
    dim_name = col.replace('sys_', '').replace('_', ' ').capitalize()
    ax.set_title(f"Evolution {dim_name} — Lit {bed}", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, ls=":", alpha=.4)
    fig.tight_layout(pad=1.2)
    return fig


def plot_radar_chart(last_row, sys_cols):
    """Diagramme en étoile des scores sous-domaines corrigés (× score global)."""
    global_score = last_row.get("score_sévérité")
    if global_score is None or pd.isna(global_score):
        return None

    labels, values = [], []
    for col in sys_cols:
        raw = last_row.get(col)
        if raw is None or pd.isna(raw):
            continue
        corrected = float(raw) * float(global_score)
        name = col.replace("sys_", "").replace("_", " ").upper()
        labels.append(f"{name}\n({corrected:.2f})")
        values.append(corrected)

    if len(labels) < 2:
        return None

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    values_closed = values + [values[0]]

    if global_score >= 0.75:
        fill_color, line_color = "#ffebee", "#c62828"
    elif global_score >= 0.50:
        fill_color, line_color = "#fff3e0", "#f57c00"
    else:
        fill_color, line_color = "#e8f5e9", "#2e7d32"

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, color=line_color, linewidth=2)
    ax.fill(angles_closed, values_closed, color=fill_color, alpha=0.55)

    # Cercles de seuil
    theta = np.linspace(0, 2 * np.pi, 200)
    for thresh, col_thresh in [(0.75, "#c62828"), (0.50, "#f57c00")]:
        ax.plot(theta, [thresh] * 200, color=col_thresh, linewidth=0.8, linestyle="--", alpha=0.45)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=8, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=7, color="#aaa")
    ax.set_title(f"Score global : {float(global_score):.2f}", pad=20, fontsize=10, fontweight="bold", color=line_color)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_vector_table(vec: dict, source_label: str = "", expanded: bool = True):
    """Affiche les features clinique sous forme de tableau HTML."""
    if not vec:
        return
    _ROWS = [
        ("Hémodynamique", [
            ("PAM",  "pam",           "mmHg"),
            ("PAD",  "pad",           "mmHg"),
            ("FC",   "heart_rate",    "bpm"),
            ("NAD",  "nad_dose_poids","µg/kg/min"),
        ]),
        ("Respiratoire", [
            ("SpO2", "spo2",         "%"),
            ("FiO2", "fio2_corr",    "%"),
            ("Ventilation","is_ventilated", ""),
        ]),
        ("Neurologique", [
            ("Conscient", "is_conscious",  ""),
            ("Sédaté",    "is_sedated",    ""),
            ("Non alerte","is_not_alert",  ""),
        ]),
        ("Général", [
            ("Âge",      "age",      "ans"),
            ("Glycémie", "glyc_cap", "mmol/L"),
        ]),
        ("Milieu intérieur", [
            ("Créatine",  "creat",        "µmol/L"),
            ("Abscence Dialyse", "abs_dialyse",   ""),
            ("Dialyse HDI",  "dialyse_hdi",   ""),
            ("Dialyse CVVHF","dialyse_cvvhf", ""),
        ]),
        ("Hémostase", [
            ("Plaquettes", "num_plq", "G/L"),
            ("Prothrombine",    "tp",      "%"),
        ]),
        ("Hépatique", [
            ("Bilirubine", "bili_tot", "µmol/L"),
        ]),
    ]
    _BIN = {"is_ventilated","is_conscious","is_sedated","is_not_alert",
            "abs_dialyse","dialyse_hdi","dialyse_cvvhf"}

    rows_html = ""
    for sys_name, features in _ROWS:
        cells = ""
        for label, key, unit in features:
            if key not in vec:
                continue
            v = vec[key]
            if key in _BIN:
                val_str = "<span style='color:#16a34a;font-weight:600'>Oui</span>" if v else "<span style='color:#94a3b8'>Non</span>"
            else:
                val_str = f"<b>{v:.1f}</b> <span style='color:#64748b;font-size:1em'>{unit}</span>"
            cells += f"""<td style='padding:3px 10px;border:1px solid #e2e8f0;white-space:nowrap;background:#ffffff'>
                <div style='font-size:1em;color:#64748b'>{label}</div>
                <div style='font-size:1em;color:#1e293b'>{val_str}</div>
            </td>"""
        if cells:
            rows_html += f"""<tr>
                <td style='padding:3px 10px;border:1px solid #e2e8f0;font-size:1.1em;
                    color:#475569;font-weight:600;white-space:nowrap;background:#f1f5f9'>{sys_name}</td>
                {cells}
            </tr>"""

    title = f"Features injectées" + (f" — {source_label}" if source_label else "")
    html = f"""<div style='overflow-x:auto;margin-top:4px'>
        <table style='border-collapse:collapse;font-size:0.85em;width:100%;
            background:#ffffff;color:#1e293b;border-radius:6px;overflow:hidden;border:1px solid #e2e8f0'>
        {rows_html}
        </table></div>"""

    with st.expander(title, expanded=expanded):
        st.markdown(html, unsafe_allow_html=True)

def render_detail_panel(bed, active_encounters, df_valid=None):
    st.markdown("---")

    # Identification du patient
    enc_id = active_encounters.get(bed)
    if not enc_id:
        df_temp = load_patient(bed, _version=st.session_state.cache_version)
        if not df_temp.empty and "encounterId" in df_temp.columns:
            enc_id = df_temp["encounterId"].iloc[-1]
    
    enc_badge = f"| EncounterId n° {enc_id}"

    # En-tête + Boutons
    c_title, c_btn1, c_btn2 = st.columns([6.5, 2, 1.3], gap="small")
    with c_title:
        st.markdown(f"#### Analyse détaillée — Lit {bed} {enc_badge}")
    with c_btn1:
        if st.button("Calculer", key="run_hist", width='stretch'):
            st.session_state.pending_run = bed
            st.rerun()
    with c_btn2:
        if st.button("✕", key="close_panel", width='stretch'):
            st.session_state.selected_bed = None
            st.rerun()

    if bed not in active_encounters:
        st.warning("Ce lit n'est plus occupé par un patient actif.")

    # Sélection heure passée (reglette)
    _saved_key = f"saved_hour_{bed}"
    _exp_key   = f"exp_open_{bed}"
    if _saved_key not in st.session_state: st.session_state[_saved_key] = 0
    if _exp_key not in st.session_state: st.session_state[_exp_key] = False
    with st.expander("Prédiction à une heure passée", expanded=st.session_state[_exp_key]):
        st.session_state[_exp_key] = True
        now = datetime.now()
        col_slider, col_info, col_run = st.columns([4, 3, 2], gap="small")
        with col_slider:
            hour_offset = st.slider("Heures en arrière", 0, 100, value=st.session_state[_saved_key], key=f"slider_{bed}", label_visibility="collapsed")
        with col_info:
            target_dt = now - __import__("datetime").timedelta(hours=hour_offset)
            st.metric("Heure ciblee", target_dt.strftime("%d/%m %H:%M"))
        with col_run:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Calculer", key=f"run_hour_{bed}", width='stretch'):
                _h = st.session_state.get(f"slider_{bed}", hour_offset)
                st.session_state[_saved_key] = _h
                st.session_state.pending_hour_run = (bed, _h)
                st.rerun()
        if st.session_state.hour_score_result is not None and st.session_state.hour_score_label == bed:
            # Récup des couleurs dynamiques
            bg_col, txt_col = get_score_colors(st.session_state.hour_score_result)
            
            st.markdown(f"""
                <div style="padding:10px; background:{bg_col}; border-left:5px solid {txt_col}; border-radius:4px; margin-top:10px; margin-bottom:10px;">
                    <small style="color:{txt_col}; font-weight:bold;">
                        Résultat H-{st.session_state.hour_score_h} ({st.session_state.hour_score_time})
                    </small><br>
                    <b style="font-size:1.2rem; color:{txt_col};">
                        Score : {st.session_state.hour_score_result:.4f}
                    </b>
                </div>
            """, unsafe_allow_html=True)

            _vec = st.session_state.get("hour_score_vector", {})
            render_vector_table(_vec, source_label="réglette", expanded=False)

    # Vecteur clinique 
    _vec_main = st.session_state.get("main_score_vector", {})
    _vec_hour = st.session_state.get("hour_score_vector", {}) if st.session_state.get("hour_score_label") == bed else {}
    _vec_show = _vec_hour or _vec_main
    _src_label = "heure choisie le " if _vec_hour else "dernier calcul le " f"{ts}"

    render_vector_table(_vec_show, source_label=_src_label, expanded=True)

    # Chargement des données
    df_hist = load_patient(bed, _version=st.session_state.cache_version)
    if df_hist.empty:
        st.info("Aucun historique disponible.")
        return

    last_row = df_hist.iloc[-1]

    # Identification des colonnes
    sys_cols = [c for c in df_hist.columns if c.startswith('sys_') and df_hist[c].notna().any()
                and c.lower().replace('sys_', '').strip() not in ('global', 'score')]

    # Radar chart (snapshot courant, scores corrigés)
    fig_radar = plot_radar_chart(last_row, sys_cols)
    if fig_radar:
        rc, _ = st.columns([1, 1])
        with rc:
            st.pyplot(fig_radar)
        plt.close(fig_radar)

    target_col = render_styled_selector(last_row, sys_cols)

    # Affichage courbe et tableau
    gc, tc = st.columns([3, 2])
    with gc:
        st.pyplot(plot_trajectory(df_hist, bed, col=target_col))
    with tc:
        disp = df_hist.sort_values("date_calcul", ascending=False).copy()
        # Calcul Delta
        disp["Δ"] = disp[target_col].diff(-1).mul(-1).round(4)
        disp = disp[["date_calcul", target_col, "Δ"]].rename(columns={"date_calcul": "Heure", target_col: "Score"})
        disp["Heure"] = disp["Heure"].dt.strftime("%H:%M")
        st.dataframe(disp, width='stretch', hide_index=True)

    st.markdown("---")

def render_styled_selector(last_row, sys_cols):
    """Génère un sélecteur de dimensions sous forme de segments colorés (Pills)."""

    global_score = last_row.get("score_sévérité")
    global_val = float(global_score) if pd.notna(global_score) else None
    options_map = {}

    g_icon = "🔴" if (global_val is not None and global_val >= 0.75) else "🟠" if (global_val is not None and global_val >= 0.50) else "🟢"
    g_label = f"{g_icon} SEVERITE ({global_val:.2f})" if global_val is not None else "🔵 GLOBAL"
    options_map[g_label] = "score_sévérité"

    # Labels systèmes physiologiques — scores corrigés (× score global)
    for col in sys_cols:
        v_raw = last_row.get(col)
        name = col.replace('sys_', '').replace('_', ' ').upper()
        if pd.notna(v_raw) and global_val is not None:
            v = float(v_raw) * global_val
            icon = "🔴" if v >= 0.75 else "🟠" if v >= 0.50 else "🟢"
            label = f"{icon} {name} ({v:.2f})"
        else:
            icon = "🟢"
            label = f"{icon} {name}"
        options_map[label] = col

    st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#475569; margin-bottom:10px; margin-top:15px;'>SYSTÈME ANALYSÉ</p>", unsafe_allow_html=True)
    
    selected_label = st.segmented_control(
        label="Dimension Selection",
        options=list(options_map.keys()),
        default=list(options_map.keys())[0],
        label_visibility="collapsed",
        key=f"seg_{last_row['lit']}"
    )
    
    return options_map.get(selected_label, "score_sévérité")

# LAYOUT PRINCIPAL

h1, h2, h3, h4 = st.columns([3.5, 1, 1.2, 1.5])
h1.title("Etat du service de réanimation")
with h2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⟳ Rafraîchir"):
        st.session_state.cache_version += 1; st.rerun()
with h3:
    st.markdown("<div style='font-size:0.78rem;color:#64748b;margin-bottom:2px'>Décalage (heures)</div>", unsafe_allow_html=True)
    global_hour = st.number_input("Décalage (heures)", min_value=0, value=0, step=1,
                                   key="global_hour_input", label_visibility="collapsed")
    
    from datetime import datetime, timedelta
    if global_hour == 0:
        _label = "Temps réel (H0)"
    else:
        date_visee = (datetime.now() - timedelta(hours=int(global_hour))).strftime("%d/%m %H:%M")
        _label = f"{date_visee} (H-{int(global_hour)})"
        
    st.markdown(f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:2px;font-style:italic'>{_label}</div>", unsafe_allow_html=True)
with h4:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶ Balayage du service"):
        label_h = f" H-{global_hour}" if global_hour else ""
        with st.spinner(f"Prédiction{label_h} en cours…"):
            ok, log = run_pipeline(hour_offset=int(global_hour))
            st.session_state.last_execution_log = log
        (st.success if ok else st.error)("Terminé" if ok else "Erreur")
        if not ok: st.expander("Logs").code(log[-600:])
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        _ref_utc = _dt.now(_tz.utc).replace(tzinfo=None) - _td(hours=int(global_hour))
        _ref_local = (_ref_utc + (_dt.now() - _dt.utcnow())).strftime("%d/%m %H:%M")
        st.session_state.last_scan_hour = int(global_hour)
        st.session_state.last_scan_ref  = _ref_local
        st.session_state.last_scan_utc  = _ref_utc if int(global_hour) > 0 else None
        st.session_state.cache_version += 1; st.rerun()

st.divider()

# ── Handlers actions asynchrones ──────────────────────────────
if st.session_state.pending_run:
    bed_to_run = st.session_state.pending_run
    st.session_state.pending_run = None
    with st.spinner(f"Recalcul de l'historique pour le lit {bed_to_run}…"):
        ok, log = run_pipeline(bed=bed_to_run, mode="history")
        st.session_state.last_execution_log = log
    if ok:
        st.success(f"✔ Historique recalculé — Lit {bed_to_run}")
    else:
        st.error(f"✗ Échec du recalcul pour le lit {bed_to_run}")
    _main_vec = {}
    _lines = log.splitlines()
    for _i, _line in enumerate(_lines):
        if "[DEBUG]" in _line:
            _main_vec = {} 
            for _dl in _lines[_i+1:]:
                if "│" not in _dl and ":" not in _dl:
                    break
                for _pair in _dl.split("│"):
                    _pair = _pair.strip()
                    if ":" in _pair:
                        try:
                            _k, _v = _pair.split(":", 1)
                            _main_vec[_k.strip()] = float(_v.strip())
                        except Exception:
                            pass
    st.session_state.main_score_vector = _main_vec
    with st.expander("Logs du calcul", expanded=not ok):
        st.code(log[-1500:] if log else "(aucun log)")
    st.session_state.cache_version += 1
    st.rerun()

if st.session_state.pending_hour_run:
    bed_h, h_off = st.session_state.pending_hour_run
    st.session_state.pending_hour_run = None
    with st.spinner(f"Prediction H-{h_off} pour le lit {bed_h}..."):
        ok, log, score, parsed_h, parsed_time, vector = run_pipeline_at_hour(bed_h, h_off)
        st.session_state.last_execution_log = log
    st.session_state.hour_score_result = score
    st.session_state.hour_score_label  = bed_h
    st.session_state.hour_score_h      = parsed_h
    st.session_state.hour_score_time   = parsed_time
    st.session_state.hour_score_vector = vector
    if ok:
        msg = f"H-{parsed_h} ({parsed_time}) score={score:.4f}" if score is not None else "Calcul termine"
        st.success(msg)
    else:
        st.error(f"Echec H-{h_off} pour {bed_h}")
        with st.expander("Logs", expanded=True): st.code(log[-1200:] if log else "(vide)")
    st.session_state.cache_version += 1
    st.rerun()

df_global = load_global(_version=st.session_state.cache_version)

with st.spinner("Connexion à la base de données…"):
    active_encounters = get_active_encounters(
        reference_utc=st.session_state.last_scan_utc,
        _version=st.session_state.cache_version
    )
    all_service_beds = get_all_service_beds()

if not active_encounters and not all_service_beds and df_global.empty:
    st.error("Impossible de joindre la base de données SQL. Vérifiez la connexion réseau.")
    st.stop()
if all_service_beds:
    beds = all_service_beds
else:
    beds = sorted(df_global["lit"].dropna().unique().tolist(),
                  key=lambda b: int(''.join(filter(str.isdigit, str(b))) or 0))

# Derniers scores valides par encounterId
if not df_global.empty and active_encounters:
    active_ids = {str(eid) for eid in active_encounters.values()}
    df_valid = df_global[df_global["encounterId"].astype(str).isin(active_ids)]
else:
    df_valid = df_global.copy()


occupied_beds = set(active_encounters.keys()) if active_encounters else set(beds)
if not df_valid.empty:
    last_all = df_valid.sort_values("date_calcul").groupby("lit")["score_sévérité"].last()
    
    occ = len(occupied_beds)
    crit = int((last_all >= .75).sum())
    warn = int(((last_all >= .50) & (last_all < .75)).sum())
    stab = int((last_all < .50).sum())

    st.markdown(f"""
    <div style="display: flex; gap: 15px; text-align: center; margin-bottom: 20px;">
        <div style="flex: 1; padding: 12px; border-radius: 6px; background: #f8f9fa; border: 1px solid #e0e0e0; border-top: 4px solid #6c757d;">
            <div style="font-size: 0.85rem; color: #555; text-transform: uppercase; font-weight: 600;">Lits occupés</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #333;">{occ}</div>
        </div>
        <div style="flex: 1; padding: 12px; border-radius: 6px; background: #ffebee; border: 1px solid #ffcdd2; border-top: 4px solid #c62828;">
            <div style="font-size: 0.85rem; color: #c62828; text-transform: uppercase; font-weight: 600;">Critique (≥ 0.75)</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #c62828;">{crit}</div>
        </div>
        <div style="flex: 1; padding: 12px; border-radius: 6px; background: #fff3e0; border: 1px solid #ffe0b2; border-top: 4px solid #f57c00;">
            <div style="font-size: 0.85rem; color: #f57c00; text-transform: uppercase; font-weight: 600;">Vigilance (0.50–0.75)</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #f57c00;">{warn}</div>
        </div>
        <div style="flex: 1; padding: 12px; border-radius: 6px; background: #e8f5e9; border: 1px solid #c8e6c9; border-top: 4px solid #2e7d32;">
            <div style="font-size: 0.85rem; color: #2e7d32; text-transform: uppercase; font-weight: 600;">Stables (&lt; 0.50)</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #2e7d32;">{stab}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

if not beds:
    st.warning("Aucun lit trouvé.")
    st.info("Causes possibles : connexion SQL indisponible, ou aucun patient présent. Vérifiez la connexion puis cliquez sur **▶ Balayage du service**.")
    st.stop()
if st.session_state.get("last_execution_log"):
    with st.expander("Features injectées", expanded=False):
        full_log = st.session_state.last_execution_log
        filtered_lines = [line for line in full_log.splitlines() 
                          if "Score" in line or "[DEBUG]" in line or "│" in line]
        
        if filtered_lines:
            st.code("\n".join(filtered_lines), language="plaintext")
        else:
            st.info("Aucune donnée d'injection détaillée dans le log actuel.")

st.subheader("État des lits")

COLS  = 6
selected = st.session_state.selected_bed
rows  = [beds[i:i+COLS] for i in range(0, len(beds), COLS)]

for row_beds in rows:
    cols = st.columns(COLS)
    for col, bed in zip(cols, row_beds):
        is_occupied = bed in active_encounters if active_encounters else True

        sub = df_valid[df_valid["lit"] == bed].sort_values("date_calcul") if is_occupied else pd.DataFrame()
        score = float(sub.iloc[-1]["score_sévérité"]) if not sub.empty else None
        ts    = sub.iloc[-1]["date_calcul"].strftime("%d/%m %H:%M") if not sub.empty else "–"

        val, cls, lbl = badge(score) if is_occupied else ("–", "none", "VIDE")
        badge_cls = cls if is_occupied else "vide"
        _h = st.session_state.last_scan_hour
        _ref = st.session_state.last_scan_ref
        hour_label = f"{_ref}" if _ref and not sub.empty else "&nbsp;"

        is_sel = selected == str(bed)
        border = "2px solid #1565c0" if is_sel else "1px solid #e0e0e0"
        bg     = "#eef2fb" if is_sel else ("#f8f9fa" if is_occupied else "#fafafa")
        opacity = "1" if is_occupied else "0.45"

        with col:
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:10px;
                        padding:12px 8px;text-align:center;opacity:{opacity};
                        height:115px;display:flex;flex-direction:column;justify-content:center;">
              <div style="font-size:.72rem;color:#999;font-family:monospace;
                          margin-bottom:2px">LIT {bed}</div>
              <div class="bed-score {cls}">{val}</div>
              <div><span class="badge badge-{badge_cls}">{lbl}</span></div>
              <div style="font-size:.62rem;color:#888;margin-top:4px;font-family:monospace;">
                {hour_label}
              </div>
            </div>""", unsafe_allow_html=True)

            btn_label = "✕ Fermer" if is_sel else "Historique"
            btn_disabled = not is_occupied and not load_patient(bed, _version=st.session_state.cache_version).shape[0]
            if st.button(btn_label, key=f"btn_{bed}", width='stretch', disabled=btn_disabled):
                if is_sel:
                    st.session_state.selected_bed = None
                else:
                    # Ouverture du panneau ET lancement du calcul 
                    st.session_state.selected_bed = str(bed)
                    st.session_state.pending_run = str(bed) 
                
                st.session_state.run_logs = ""
                st.session_state.run_ok = None
                st.session_state.hour_score_result = None
                st.session_state.hour_score_label = ""
                st.rerun()

    # Panneau sous la rangée du lit sélectionné
    if selected and selected in [str(b) for b in row_beds]:
        render_detail_panel(selected, active_encounters, df_valid)