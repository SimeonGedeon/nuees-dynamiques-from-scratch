import numpy as np
import matplotlib.pyplot as plt
from nuees_dynamiques import NueesDynamiques

def afficher_menu():
    print("====================================================")
    print("  TP : ALGORITHME DES NUÉES DYNAMIQUES (FROM SCRATCH)")
    print("====================================================\n")
    print("Sélectionnez le type de noyau pour la représentation des nuées :")
    print("1. Un point (Ramène à l'algorithme K-Means)")
    print("2. Un ensemble de points représentatifs")
    print("3. Des axes factoriels")
    print("4. Une distribution de probabilité")
    print("5. Une structure représentative")

def main():
    afficher_menu()
    choix = input("\nEntrez votre choix (1-5) : ").strip()

    map_noyaux = {
        "1": "point",
        "2": "ensemble_points",
        "3": "axe_factoriel",
        "4": "distribution",
        "5": "structure_representative"
    }

    type_choisi = map_noyaux.get(choix, "point")
    print(f"\n[INFO] Structure de noyau sélectionnée : '{type_choisi}'")

    if type_choisi == "point":
        # Génération d'un jeu de données synthétique 2D (300 points)
        np.random.seed(42)
        c1 = np.random.randn(100, 2) + np.array([4, 4])
        c2 = np.random.randn(100, 2) + np.array([-4, -3])
        c3 = np.random.randn(100, 2) + np.array([4, -4])
        X = np.vstack([c1, c2, c3])

        # Instanciation et entraînement
        model = NueesDynamiques(k=3, type_noyau='point', seed=42)
        model.fit(X)

        print(f"[RÉSULTAT] Convergence atteinte en {len(model.inertia_history)} itérations.")
        print(f"[RÉSULTAT] Inertie intra-classe finale W = {model.inertia_history[-1]:.4f}")

        # --- Visualisations ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Graphique 1 : Partitionnement
        axes[0].scatter(X[:, 0], X[:, 1], c=model.labels, cmap='viridis', alpha=0.6, edgecolors='k')
        axes[0].scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', marker='X', s=200, label='Centroïdes')
        axes[0].set_title("Partitionnement obtenu (Nuées Dynamiques)")
        axes[0].legend()
        axes[0].grid(True)

        # Graphique 2 : Courbe de convergence
        axes[1].plot(range(1, len(model.inertia_history) + 1), model.inertia_history, marker='o', color='b')
        axes[1].set_xlabel("Itérations")
        axes[1].set_ylabel("Inertie intra-classe (W)")
        axes[1].set_title("Décroissance de l'inertie intra-classe")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("resultat_nuees_dynamiques.png")
        print("[INFO] Graphique enregistré sous 'resultat_nuees_dynamiques.png'.")
        plt.show()

    else:
        print(f"[INFO] L'algorithme a pris en compte votre choix '{type_choisi}'.")
        print("[INFO] Les métriques personnalisées pour ce noyau sont en cours de développement.")

if __name__ == "__main__":
    main()