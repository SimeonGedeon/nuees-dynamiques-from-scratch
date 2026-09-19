"""
Version web de la démo (Streamlit).

Usage :
    streamlit run app_streamlit.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src import NueesDynamiques, NoyauPoint, NoyauMultiPoints

st.set_page_config(page_title="Nuées Dynamiques", layout="wide")
st.title("Démonstration — Algorithme des Nuées Dynamiques")

# ----- Barre latérale -----
st.sidebar.header("Paramètres")
dataset = st.sidebar.radio("Jeu de données", ["Blobs", "Moons"])
k = st.sidebar.slider("Nombre de classes K", 2, 8, 4)
noyau = st.sidebar.radio("Type de noyau", ["Point (K-means)", "Multi-points"])
p = st.sidebar.slider("Nombre de prototypes p", 2, 10, 3)
seed = st.sidebar.number_input("Graine aléatoire", 0, 999, 0)

# ----- Données -----
if dataset == "Blobs":
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=0)
else:
    X, _ = make_moons(n_samples=400, noise=0.08, random_state=0)

# ----- Modèle -----
if noyau == "Point (K-means)":
    fabrique = lambda: NoyauPoint()
else:
    fabrique = lambda: NoyauMultiPoints(p=p)

model = NueesDynamiques(k=k, noyau_factory=fabrique, seed=seed).fit(X)

# ----- Affichage -----
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=model.labels, cmap="tab10", s=12)
    for g in model.noyaux:
        if hasattr(g, "point") and g.point is not None:
            ax.scatter(*g.point, c="red", marker="X", s=200, edgecolors="black")
    ax.set_title(f"Clusters — K={k}")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(model.inertia_history, marker="o")
    ax.set_xlabel("Itération")
    ax.set_ylabel("W")
    ax.set_title("Convergence")
    st.pyplot(fig)

st.metric("Inertie finale W", f"{model.inertia_history[-1]:.2f}")
st.metric("Itérations", len(model.inertia_history))