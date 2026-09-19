"""
Point d'entrée du projet Nuées Dynamiques.

Usage :
    python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src import NueesDynamiques, NoyauPoint, NoyauMultiPoints
from sklearn.datasets import make_blobs


def demo():
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=0)

    print("=== Noyau point (K-means) ===")
    m1 = NueesDynamiques(k=4, noyau_factory=lambda: NoyauPoint(), seed=0)
    m1.fit(X)
    print(f"W final = {m1.inertia_history[-1]:.2f}")
    print(f"Itérations = {len(m1.inertia_history)}")

    print("\n=== Noyau multi-points (p=3) ===")
    m2 = NueesDynamiques(k=4, noyau_factory=lambda: NoyauMultiPoints(p=3), seed=0)
    m2.fit(X)
    print(f"W final = {m2.inertia_history[-1]:.2f}")
    print(f"Itérations = {len(m2.inertia_history)}")


if __name__ == "__main__":
    demo()