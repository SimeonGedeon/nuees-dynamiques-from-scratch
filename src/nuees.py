import numpy as np
from .noyaux import NoyauPoint


class NueesDynamiques:
    """Algorithme des nuées dynamiques (Diday, 1971)."""

    def __init__(self, k=3, noyau_factory=None,
                 max_iter=100, tol=1e-4, seed=42, verbose=False):
        self.k = k
        self.noyau_factory = noyau_factory or (lambda: NoyauPoint())
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.verbose = verbose
        self.noyaux = []
        self.labels = None
        self.inertia_history = []
        self.n_iter_ = 0

    def _affecter(self, X):
        D = np.column_stack([g.distance(X) for g in self.noyaux])
        return np.argmin(D, axis=1), D

    def _representer(self, X, labels):
        nouveaux = []
        for j in range(self.k):
            Xj = X[labels == j]
            g = self.noyau_factory()
            if len(Xj) == 0:
                g.init(X)
            else:
                g.init(Xj)
            nouveaux.append(g)
        return nouveaux

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)

        self.noyaux = []
        for _ in range(self.k):
            g = self.noyau_factory()
            idx = rng.choice(len(X), size=min(2, len(X)), replace=False)
            g.init(X[idx])
            self.noyaux.append(g)

        self.inertia_history = []

        for it in range(self.max_iter):
            labels, D = self._affecter(X)
            critere = float(np.sum(np.min(D, axis=1) ** 2))
            self.inertia_history.append(critere)

            if self.verbose:
                print(f"itération {it+1:3d} | W = {critere:.4f}")

            self.noyaux = self._representer(X, labels)

            nouveaux_labels, _ = self._affecter(X)
            if np.array_equal(labels, nouveaux_labels):
                break

        self.labels = labels
        self.n_iter_ = it + 1
        return self

    def predict(self, X):
        labels, _ = self._affecter(np.asarray(X, float))
        return labels