"""
Interface graphique Tkinter pour l'algorithme des Nuées Dynamiques.

Usage :
    python app.py
"""

import sys
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.datasets import make_blobs, make_moons

sys.path.insert(0, os.path.dirname(__file__))
from src import NueesDynamiques, NoyauPoint, NoyauMultiPoints


# =========================================================
class AppNueesDynamiques(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nuées Dynamiques — Démonstration interactive")
        self.geometry("1200x700")
        self.configure(bg="#f0f0f0")

        # ----- État -----
        self.X = None
        self.modele = None

        # ----- Construction de l'interface -----
        self._build_panneau_gauche()
        self._build_panneau_droit()

        # Charger un jeu par défaut
        self._charger_donnees()

    # -----------------------------------------------------
    def _build_panneau_gauche(self):
        cadre = ttk.LabelFrame(self, text="Paramètres", padding=15)
        cadre.place(x=10, y=10, width=320, height=680)

        # Choix du jeu de données
        ttk.Label(cadre, text="Jeu de données :").pack(anchor="w", pady=(0, 5))
        self.var_dataset = tk.StringVar(value="blobs")
        ttk.Radiobutton(cadre, text="Blobs (classes sphériques)",
                        variable=self.var_dataset, value="blobs",
                        command=self._charger_donnees).pack(anchor="w")
        ttk.Radiobutton(cadre, text="Moons (lunes entrelacées)",
                        variable=self.var_dataset, value="moons",
                        command=self._charger_donnees).pack(anchor="w")

        ttk.Separator(cadre, orient="horizontal").pack(fill="x", pady=15)

        # Nombre de classes K
        ttk.Label(cadre, text="Nombre de classes K :").pack(anchor="w")
        self.var_k = tk.IntVar(value=4)
        ttk.Scale(cadre, from_=2, to=8, orient="horizontal",
                  variable=self.var_k).pack(fill="x")
        self.lbl_k = ttk.Label(cadre, text="K = 4")
        self.lbl_k.pack(anchor="w")
        self.var_k.trace_add("write",
                             lambda *_: self.lbl_k.config(
                                 text=f"K = {self.var_k.get()}"))

        ttk.Separator(cadre, orient="horizontal").pack(fill="x", pady=15)

        # Type de noyau
        ttk.Label(cadre, text="Type de noyau :").pack(anchor="w", pady=(0, 5))
        self.var_noyau = tk.StringVar(value="point")
        ttk.Radiobutton(cadre, text="Point (= K-means)",
                        variable=self.var_noyau, value="point").pack(anchor="w")
        ttk.Radiobutton(cadre, text="Multi-points",
                        variable=self.var_noyau, value="multi").pack(anchor="w")

        # Nombre de prototypes (si multi)
        ttk.Label(cadre, text="Nombre de prototypes p :").pack(anchor="w",
                                                               pady=(10, 0))
        self.var_p = tk.IntVar(value=3)
        ttk.Spinbox(cadre, from_=2, to=10, textvariable=self.var_p,
                    width=5).pack(anchor="w")

        ttk.Separator(cadre, orient="horizontal").pack(fill="x", pady=15)

        # Initialisation
        ttk.Label(cadre, text="Graine aléatoire :").pack(anchor="w")
        self.var_seed = tk.IntVar(value=0)
        ttk.Spinbox(cadre, from_=0, to=999, textvariable=self.var_seed,
                    width=5).pack(anchor="w")

        ttk.Separator(cadre, orient="horizontal").pack(fill="x", pady=15)

        # Boutons
        ttk.Button(cadre, text="▶ Lancer",
                   command=self._lancer).pack(fill="x", pady=5)
        ttk.Button(cadre, text="⟳ Nouvelles données",
                   command=self._charger_donnees).pack(fill="x", pady=5)
        ttk.Button(cadre, text="✖ Quitter",
                   command=self.destroy).pack(fill="x", pady=5)

        # Zone de résultats
        ttk.Label(cadre, text="Résultats :").pack(anchor="w", pady=(20, 5))
        self.txt_resultats = tk.Text(cadre, height=8, width=35,
                                     font=("Consolas", 9))
        self.txt_resultats.pack(fill="both", expand=True)

    # -----------------------------------------------------
    def _build_panneau_droit(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            1, 2, figsize=(9, 5))
        self.fig.tight_layout(pad=3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().place(x=350, y=10, width=830, height=680)

    # -----------------------------------------------------
    def _charger_donnees(self):
        if self.var_dataset.get() == "blobs":
            self.X, _ = make_blobs(n_samples=500, centers=4,
                                   cluster_std=0.8, random_state=0)
        else:
            self.X, _ = make_moons(n_samples=400, noise=0.08,
                                   random_state=0)

        # Effacer les résultats précédents
        self.txt_resultats.delete("1.0", tk.END)
        self._afficher_donnees_brutes()

    # -----------------------------------------------------
    def _afficher_donnees_brutes(self):
        self.ax1.clear()
        self.ax1.scatter(self.X[:, 0], self.X[:, 1],
                         s=12, c="steelblue", alpha=0.7)
        self.ax1.set_title("Données brutes")
        self.ax1.grid(alpha=0.3)

        self.ax2.clear()
        self.ax2.text(0.5, 0.5, "Cliquez sur « Lancer »",
                      ha="center", va="center", fontsize=14, color="gray")
        self.ax2.axis("off")

        self.canvas.draw()

    # -----------------------------------------------------
    def _lancer(self):
        try:
            k = self.var_k.get()
            seed = self.var_seed.get()
            noyau_type = self.var_noyau.get()
            p = self.var_p.get()

            if noyau_type == "point":
                fabrique = lambda: NoyauPoint()
                nom_noyau = "Point (K-means)"
            else:
                fabrique = lambda: NoyauMultiPoints(p=p)
                nom_noyau = f"Multi-points (p={p})"

            self.modele = NueesDynamiques(k=k, noyau_factory=fabrique,
                                          seed=seed)
            self.modele.fit(self.X)

            self._afficher_resultats(nom_noyau)

        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    # -----------------------------------------------------
    def _afficher_resultats(self, nom_noyau):
        # ----- Graphique 1 : clusters -----
        self.ax1.clear()
        self.ax1.scatter(self.X[:, 0], self.X[:, 1],
                         c=self.modele.labels, cmap="tab10",
                         s=12, alpha=0.8)
        # Centroïdes (noyau point) ou prototypes (multi-points)
        for g in self.modele.noyaux:
            if hasattr(g, "point") and g.point is not None:
                self.ax1.scatter(*g.point, c="red", marker="X",
                                 s=200, edgecolors="black")
            elif hasattr(g, "points") and g.points is not None:
                self.ax1.scatter(g.points[:, 0], g.points[:, 1],
                                 c="red", marker="X",
                                 s=120, edgecolors="black")
        self.ax1.set_title(f"Clusters — {nom_noyau}")
        self.ax1.grid(alpha=0.3)

        # ----- Graphique 2 : convergence -----
        self.ax2.clear()
        self.ax2.plot(self.modele.inertia_history, marker="o",
                      color="steelblue")
        self.ax2.set_xlabel("Itération")
        self.ax2.set_ylabel("Inertie intra-classe W")
        self.ax2.set_title("Convergence")
        self.ax2.grid(alpha=0.3)

        self.canvas.draw()

        # ----- Résumé texte -----
        W_final = self.modele.inertia_history[-1]
        n_iter = len(self.modele.inertia_history)

        self.txt_resultats.delete("1.0", tk.END)
        self.txt_resultats.insert(tk.END,
            f"Noyau       : {nom_noyau}\n"
            f"K           : {self.var_k.get()}\n"
            f"Graine      : {self.var_seed.get()}\n"
            f"Itérations  : {n_iter}\n"
            f"W final     : {W_final:.2f}\n"
            f"W initial   : {self.modele.inertia_history[0]:.2f}\n"
            f"Baisse      : "
            f"{(1 - W_final / self.modele.inertia_history[0]) * 100:.1f} %\n"
        )


# =========================================================
if __name__ == "__main__":
    app = AppNueesDynamiques()
    app.mainloop()