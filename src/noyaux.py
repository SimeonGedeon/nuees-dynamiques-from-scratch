import numpy as np
from abc import ABC, abstractmethod


class Noyau(ABC):
    """Représentant abstrait d'une classe (au sens de Diday)."""

    @abstractmethod
    def distance(self, X):
        ...

    @abstractmethod
    def update(self, X_classe):
        ...

    def init(self, X):
        self.update(X)
        return self


class NoyauPoint(Noyau):
    """Noyau = un point (barycentre). Cas particulier : K-means."""
    def __init__(self, point=None):
        self.point = None if point is None else np.asarray(point, float)

    def distance(self, X):
        return np.linalg.norm(X - self.point, axis=1)

    def update(self, X_classe):
        self.point = X_classe.mean(axis=0)
        return self


class NoyauMultiPoints(Noyau):
    """Noyau = p prototypes (k-means interne sur la classe)."""
    def __init__(self, p=3):
        self.p = p
        self.points = None

    def distance(self, X):
        d = np.linalg.norm(X[:, None, :] - self.points[None, :, :], axis=2)
        return d.min(axis=1)

    def update(self, X_classe):
        from sklearn.cluster import KMeans
        if len(X_classe) < self.p:
            self.points = X_classe.copy()
        else:
            km = KMeans(n_clusters=self.p, n_init=5, random_state=0).fit(X_classe)
            self.points = km.cluster_centers_
        return self


class NoyauAxe(Noyau):
    """Noyau = un axe factoriel (point + direction)."""
    def __init__(self):
        self.centre = None
        self.direction = None

    def distance(self, X):
        V = X - self.centre
        proj = V @ self.direction
        orth = V - np.outer(proj, self.direction)
        return np.linalg.norm(orth, axis=1)

    def update(self, X_classe):
        self.centre = X_classe.mean(axis=0)
        cov = np.cov((X_classe - self.centre).T)
        vals, vecs = np.linalg.eigh(cov)
        self.direction = vecs[:, -1]
        return self