import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import config as cfg


def plot_patient_trajectory(bed_label):
    """
    Génère la courbe d'évolution du score de sévérité LSTM pour un lit donné.
    """
    file_path = os.path.join(cfg.OUTPUT_DIR, f"historique_lit_{bed_label}.csv")
    
    if not os.path.exists(file_path):
        print(f" [ERREUR] Impossible de tracer la courbe : le fichier {file_path} est introuvable.")
        return
        
    try:
        df = pd.read_csv(file_path, sep=';')
        df['date_calcul'] = pd.to_datetime(df['date_calcul'])
        # Rétablissement de l'ordre chronologique
        df = df.sort_values('date_calcul')

        plt.figure(figsize=(10, 5))
    
        # Tracé de la courbe principale
        plt.plot(
            df['date_calcul'], 
            df['score_sévérité'], 
            marker='o', 
            linestyle='-', 
            color='#1f77b4', 
            linewidth=2,
            markersize=8,
            label='Score de sévérité'
        )
        
        # zone d'alerte 
        plt.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='Seuil critique théorique')
        plt.fill_between(df['date_calcul'], 0.75, 1.0, color='red', alpha=0.1)

        # axes et titre
        plt.title(f"Trajectoire Clinique LSTM - Lit {bed_label}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Heure d'évaluation", fontsize=12)
        plt.ylabel("Probabilité de détérioration (0 à 1)", fontsize=12)

        plt.ylim(-0.05, 1.05) 
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        
        # Esthétique 
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        # Affichage du graphique
        print(f" └── Génération de la courbe pour le lit {bed_label}...")
        plt.show()

    except Exception as e:
        print(f" [ERREUR GRAPHIQUE] Échec du tracé pour le lit {bed_label} : {e}")