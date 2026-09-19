# Nuées Dynamiques (Diday, 1971) — Implémentation from scratch

Implémentation en Python pur (NumPy) de l'algorithme des nuées dynamiques,
avec choix configurable du type de noyau.

## Contenu

- `src/noyaux.py` : classes `Noyau`, `NoyauPoint`, `NoyauMultiPoints`, `NoyauAxe`
- `src/nuees.py` : classe `NueesDynamiques` (alternance A / W)
- `scripts/run_demo.py` : génère les figures du rapport
- `tests/` : tests unitaires
- `notebooks/demo.ipynb` : démonstration interactive

## Installation

```bash
git clone https://github.com/<ton-user>/nuees-dynamiques.git
cd nuees-dynamiques
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Utilisation

```python
from src import NueesDynamiques, NoyauPoint
import numpy as np

X = np.random.randn(300, 2)
model = NueesDynamiques(k=3, noyau_factory=lambda: NoyauPoint(), seed=42).fit(X)
print(model.labels)
print(model.inertia_history)
```

## Générer les figures

```bash
python scripts/run_demo.py
```

Les figures sont écrites dans `figures/`.

## Tests

```bash
pytest tests/ -v
```

## Auteur

KIMBUNGU SIMÉON GEDEON — TP Sciences des données.