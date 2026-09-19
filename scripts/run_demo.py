"""
Génère toutes les figures du rapport.
Usage : python scripts/run_demo.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

# ajout du dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import NueesDynamiques, NoyauPoint, NoyauMultiPoints, NoyauAxe

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. Données synthétiques
# ---------------------------------------------------------
X_blobs, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=0)
X_moons, _ = make_moons(n_samples=400, noise=0.08, random_state=0)


# ---------------------------------------------------------
# 2. Figure : convergence de W
# ---------------------------------------------------------
model = NueesDynamiques(k=4, noyau_factory=lambda: NoyauPoint(), seed=0)
model.fit(X_blobs)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(model.inertia_history, marker="o", color="steelblue")
ax.set_xlabel("Itération")
ax.set_ylabel("Inertie intra-classe W")
ax.set_title("Convergence de l'inertie intra-classe")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "convergence.png"), dpi=150)
plt.close(fig)
print("[OK] figures/convergence.png")


# ---------------------------------------------------------
# 3. Figure : comparaison des noyaux
# ---------------------------------------------------------
model_point = NueesDynamiques(k=4, noyau_factory=lambda: NoyauPoint(), seed=0).fit(X_blobs)
model_multi = NueesDynamiques(k=4, noyau_factory=lambda: NoyauMultiPoints(p=3), seed=0).fit(X_blobs)
model_axe   = NueesDynamiques(k=4, noyau_factory=lambda: NoyauAxe(), seed=0).fit(X_blobs)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
titres = ["Noyau point (= K-means)",
          "Noyau multi-points (p=3)",
          "Noyau axe factoriel"]
modeles = [model_point, model_multi, model_axe]

for ax, m, t in zip(axes, modeles, titres):
    ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=m.labels, cmap="tab10", s=12)
    ax.set_title(t)
    ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "comparaison_noyaux.png"), dpi=150)
plt.close(fig)
print("[OK] figures/comparaison_noyaux.png")


# ---------------------------------------------------------
# 4. Figure : lunes entrelacées (multi-points)
# ---------------------------------------------------------
model_moons = NueesDynamiques(
    k=2, noyau_factory=lambda: NoyauMultiPoints(p=6), seed=0
).fit(X_moons)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X_moons[:, 0], X_moons[:, 1], c=model_moons.labels, cmap="tab10", s=15)
ax.set_title("Noyau multi-points (p=6) sur deux lunes entrelacées")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "lunes.png"), dpi=150)
plt.close(fig)
print("[OK] figures/lunes.png")


# ---------------------------------------------------------
# 5. Résumé console
# ---------------------------------------------------------
print("\n--- Résumé ---")
print(f"Noyau point      : W = {model_point.inertia_history[-1]:.2f} "
      f"en {model_point.n_iter_} itérations")
print(f"Noyau multi-points: W = {model_multi.inertia_history[-1]:.2f} "
      f"en {model_multi.n_iter_} itérations")
print(f"Noyau axe        : W = {model_axe.inertia_history[-1]:.2f} "
      f"en {model_axe.n_iter_} itérations")