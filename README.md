***Ce pipeline extrait dynamiquement les données patient, les traites et sort un score de sévérité en "direct". Le tout est affichable dans une interface streamlit qui centralise toutes les fonctionnalités du pipeline.***


## 1. Architecture du Pipeline

**Structure des modules :**
*  `config.py` : Fichier de configuration globale (chemins, base de données, chargement du thésaurus).

*  `extract.py` : Module de génération de requêtes SQL dynamiques vers la base de données ICCA.

*  `preprocessing.py` : Préparation des données incluant l'agrégation, l'imputation des valeurs manquantes et les séries temporelles.

*  `predict.py` : Module chargeant les modèles LSTM (par label et global) et calculant les probabilités de défaillance par organe.

*  `main.py` : Orchestrateur central et interface en ligne de commande (CLI).

*  `app_rea.py` : Interface web interactive destinée aux soignants, propulsée par Streamlit.



## 2. Utilisation de `main.py`

Le script `main.py` agit comme le moteur d'exécution du pipeline. Il s'opère via l'invite de commande et propose trois modes opératoires distincts :

**A. Balayage complet du service**

Cette commande évalue simultanément tous les lits occupés du service de réanimation.

```bash
python main.py
```
*Option :* L'ajout de l'argument `--w 4` permet de définir manuellement le nombre de cœurs alloués au calcul. L'ajout de `--h n` permet de réaliser un balayage complet avec un décalage temporel de $n$ heures.


**B. Analyse avec historique d'un lit spécifique**
Cette commande extrait les données et calcule l'évolution du score heure par heure sur les 6 dernières heures pour un lit donné.
```bash
python main.py 12
```
*(Le paramètre `12` correspond au numéro d'identification du lit ciblé)*.

**C. Prédiction à une heure passée ciblée**
Cette commande calcule le score de sévérité ponctuel à un instant $T-X$ heures.
```bash
python main.py 12 --h 3
```
*(Exécute le pipeline pour le lit 12 tel qu'il se présentait il y a exactement 3 heures)*.

## 3.  Dictionnaire de Données (`thesaurus.json`)
Le fichier `thesaurus.json` agit comme une table de traduction dynamique entre les concepts médicaux bruts de la base de données ICCA et les méthodes de traitements appliquées par le pipeline.

#### A. Architecture d'une variable standard
Chaque paramètre physiologique doit être déclaré sous la racine `"features"` selon la structure suivante: 

```json
    "HEART_RATE": {
      "column": "heart_rate",
      "table": "PtAssessment",
      "type": "standard",
      "label": [
        "hémodynamique"
      ],
      "codes": [
        "heartRateInt.heartRate",
        "heartRateInt.heartRate.ptAdult"
      ],
      "agg_method": "median",
      "imputation_method": "interpolate_ffill_bfill",
      "default_value": 0.0
    }

```

#### B. Paramètres obligatoires

Pour garantir l'extraction SQL et le prétraitement, les champs suivants doivent impérativement être renseignés :

* **`column`** (ou **`columns`**) : Le nom final de la ou des variables mathématiques telles qu'elles apparaissent pour le traitement par l'algorithme  (ex: `"pam"`, `"fio2_corr"`).
* **`table`** : La table SQL source dans ICCA (ex: `"PtAssessment"`, `"ptLabResult"`, `"ptMedication"`).
* **`type`** : Définit la méthode algorithmique d'extraction SQL utilisée dans `utils.py`.
	* `"standard"` : Extraction classique par `attributeId`.
	* `"categorical_mapping"` : Conversion d'états textuels (ex: statuts neurologiques) en variables booléennes (One-Hot Encoding).
	* `"drip_medication"` : Extraction de débits de perfusion continus (IVSE) avec isolation de la dose .
	* `"fio2_corrected"` : Déclenchement de la fonction mathématique de conversion du débit d'oxygène (L/min) en pourcentage de fraction inspirée.

* **`label`** : Liste définissant l'appartenance de la variable à un ou plusieurs systèmes physiologiques (ex: `["respiratoire"]`, `["hémostase", "hépatique"]`). Ce paramètre est nécessaire pour l'architecture de l'apprentissage fédéré, car il détermine dynamiquement quel sous-modèle de prédiction recevra cette donnée lors du calcul.
* **`codes`** : La liste exacte des `dictionaryPropName` ou `dictionaryLabel` présents dans ICCA.
* **`agg_method`** : La fonction mathématique utilisée pour réduire de multiples mesures survenues au cours d'une même heure en un point unique (`"median"`, `"max"`, `"sum"`, `"first"`).
* **`imputation_method`** : La stratégie de comblement des données manquantes requise pour assurer la continuité de la série temporelle avant son passage dans le réseau récurrent (`"interpolate_ffill_bfill"`, `"ffill_zero"`, `"fillna_zero"`).
* **`default_value`** : La valeur physiologique de sécurité injectée si la donnée est totalement absente de l'historique du patient (ex: `21.0` pour la FiO2, `0.0` pour la noradrénaline).


## 4.  Interface Web (`app_rea.py`)

L'interface graphique permet la présentation des résultats et l'exploration visuelle de l'outil.

**Lancement du serveur web :**
```bash
streamlit run app_rea.py
```
ou
```bash
uv run --with-requirements requirements.txt python -m streamlit run app_rea.py
```
**Fonctionnalités :**

* **Tableau de bord matriciel :** Affiche la cartographie des lits avec un code couleur indicatif de la sévérité (Vert: Stable <50%, Orange: Vigilance, Rouge: Critique >75%).
* **Balayage global :** Le bouton central "▶ Balayage du service" déclenche l'exécution en arrière-plan du script `main.py` pour tous les patients et actualise dynamiquement les scores et valeurs cliniques affichées.
* **Analyse détaillée par patient :** Un clic sur la vignette d'un lit déploie un panneau de contrôle inférieur contenant pour les 6 dernières heures:
	* **Les valeurs cliniques :** Un tableau récapitulant les données lues et interprétées par l'algorithme lors de la dernière évaluation.
	* **La réglette temporelle :** Un curseur permettant de simuler l'algorithme dans le passé (de $H-0$ à $H-100$) afin d'observer les données et scores passés pour construire une évolution de plus de 6 heures.
	* **Les trajectoires cliniques :** Les courbes d'évolution temporelle des probabilités de défaillance.
	* **Sélecteur de systèmes physiologiques :** Des vignettes permettent de basculer l'affichage de la courbe du score global vers l'évaluation prédictive spécifique d'un label *(Hémodynamique, Respiratoire, Neurologique, Général, Milieu intérieur, Hémostase, Hépatique)*.