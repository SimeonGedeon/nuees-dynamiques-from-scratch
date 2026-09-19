# Nuées Dynamiques (Diday, 1971) — Implémentation from scratch

Implémentation en Python pur (NumPy uniquement) de l'algorithme des
nuées dynamiques, avec choix configurable du type de noyau.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Présentation

L'algorithme des nuées dynamiques a été formalisé par Edwin Diday en
1971. Contrairement à la vision réductrice qui l'assimile aux
\(K\)-means, il s'agit d'un cadre général où chaque classe peut être
représentée par différents types de noyaux : un point, un ensemble de
points, un axe factoriel, une distribution de probabilité, etc.

Ce projet implémente ce cadre général from scratch, sans utiliser
scikit-learn pour l'algorithme lui-même.

---

## Structure du projet

```
nuees-dynamiques-from-scratch/
├── src/
│   ├── __init__.py
│   ├── noyaux.py            # Noyau, NoyauPoint, NoyauMultiPoints, NoyauAxe
│   └── nuees.py             # NueesDynamiques (alternance A / W)
├── scripts/
│   └── run_demo.py          # génère les figures du rapport
├── notebooks/
│   └── demo.ipynb           # démonstration interactive
├── tests/
│   └── test_nuees.py        # tests unitaires (pytest)
├── figures/                 # figures générées
├── main.py                  # point d'entrée en ligne de commande
├── gui.py                   # interface graphique Tkinter
├── app_streamlit.py         # interface web Streamlit
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/SimeonGedeon/nuees-dynamiques-from-scratch.git
cd nuees-dynamiques-from-scratch
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## Utilisation en ligne de commande

```bash
python main.py
```

Ou directement en Python :

```python
from src import NueesDynamiques, NoyauPoint
import numpy as np

X = np.random.randn(300, 2)
model = NueesDynamiques(k=3, noyau_factory=lambda: NoyauPoint(), seed=42).fit(X)

print(model.labels)              # étiquette de chaque point
print(model.inertia_history)     # évolution de l'inertie intra-classe W
```

---

## Interface graphique

### Version Tkinter (locale)

```bash
python app.py
```

Une fenêtre s'ouvre avec :
- un panneau de paramètres (jeu de données, K, noyau, graine) ;
- deux graphiques côte à côte (clusters + courbe de convergence).

### Version Streamlit (web)

```bash
pip install streamlit
streamlit run app_streamlit.py
```

La page s'ouvre sur `http://localhost:8501`.

---

## Générer les figures du rapport

```bash
python scripts/run_demo.py
```

Les figures sont écrites dans `figures/` :
- `convergence.png`
- `comparaison_noyaux.png`
- `lunes.png`

---

## Tests

```bash
pytest tests/ -v
```

Les tests vérifient notamment :
- l'équivalence avec les \(K\)-means quand le noyau est un point ;
- la décroissance monotone de l'inertie intra-classe ;
- le bon fonctionnement du noyau multi-points.

---

## Rapport LaTeX

Le rapport associé est fourni dans le dépôt :
- source : `tp_sc_data_algo_nuees_dynamiques.tex`
- version compilée : `tp_sc_data_algo_nuees_dynamiques.pdf`

---

## Auteur

**KIMBUNGU SIMÉON Gédéon** — Master 1 Informatique,
Université de Kinshasa.

## Encadreur

Doctorant **Gradi L. KAMINGU**