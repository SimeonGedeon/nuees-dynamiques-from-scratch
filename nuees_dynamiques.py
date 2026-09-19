import numpy as np

class NueesDynamiques:
    """
    Implémentation de l'algorithme généralisé des Nuées Dynamiques (E. Diday).
    Permet le choix du type de noyau/étalon pour la représentation des classes.
    """
    def __init__(self, k=3, type_noyau='point', max_iter=300, tol=1e-4, seed=42):
        """
        Parameters:
        -----------
        k : int
            Nombre de nuées/clusters.
        type_noyau : str
            Type de représentation ('point', 'ensemble_points', 'axe_factoriel',
            'distribution', 'structure_representative').
        """
        self.k = k
        self.type_noyau = type_noyau
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids = None
        self.labels = None
        self.inertia_history = []

    def _calculer_distances(self, X):
        if self.type_noyau == 'point':
            # Cas raméné au K-means : distance euclidienne aux centroïdes (N, K)
            differences = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
            return np.linalg.norm(differences, axis=2)
        else:
            # Extension possible pour d'autres types de noyaux
            raise NotImplementedError(
                f"Le type de noyau '{self.type_noyau}' nécessite une métrique spécifique."
            )

    def fit(self, X):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape

        # Aiguillage selon le choix de l'utilisateur
        if self.type_noyau == 'point':
            # --- DEROULEMENT DU K-MEANS CLASSIQUE ---
            indices_init = np.random.choice(n_samples, self.k, replace=False)
            self.centroids = X[indices_init].astype(float)

            for iteration in range(self.max_iter):
                # 1. Étape d'affectation
                distances = self._calculer_distances(X)
                self.labels = np.argmin(distances, axis=1)

                # 2. Calcul de l'inertie intra-classe
                min_distances = np.min(distances, axis=1)
                w_inertia = np.sum(min_distances ** 2)
                self.inertia_history.append(w_inertia)

                # 3. Étape de réestimation (barycentres)
                nouveaux_centroides = np.array([
                    X[self.labels == k].mean(axis=0) if np.sum(self.labels == k) > 0 else self.centroids[k]
                    for k in range(self.k)
                ])

                # 4. Test de convergence
                shift = np.linalg.norm(nouveaux_centroides - self.centroids)
                self.centroids = nouveaux_centroides

                if shift < self.tol:
                    break
        else:
            print(f"Initialisation de la nuée dynamique avec le noyau : {self.type_noyau}")

        return self