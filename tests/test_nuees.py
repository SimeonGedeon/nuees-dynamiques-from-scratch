import numpy as np
import pytest
from src import NueesDynamiques, NoyauPoint, NoyauMultiPoints


def test_kmeans_equivalent():
    """Avec NoyauPoint, on doit retrouver une partition cohérente."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (50, 2)),
                   rng.normal(5, 0.3, (50, 2))])
    model = NueesDynamiques(k=2, noyau_factory=lambda: NoyauPoint(), seed=0).fit(X)
    assert len(np.unique(model.labels)) == 2


def test_inertie_decroissante():
    """L'inertie W doit décroître (au sens large) à chaque itération."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (200, 3))
    model = NueesDynamiques(k=3, noyau_factory=lambda: NoyauPoint(), seed=0).fit(X)
    hist = model.inertia_history
    assert all(hist[i+1] <= hist[i] + 1e-9 for i in range(len(hist) - 1))


def test_predict():
    """predict doit renvoyer une étiquette par point."""
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (100, 2))
    model = NueesDynamiques(k=3, noyau_factory=lambda: NoyauPoint(), seed=0).fit(X)
    labels = model.predict(X)
    assert labels.shape == (100,)
    assert set(labels).issubset({0, 1, 2})


def test_multipoints():
    """Le noyau multi-points doit fonctionner sur des données simples."""
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (120, 2))
    model = NueesDynamiques(
        k=2, noyau_factory=lambda: NoyauMultiPoints(p=3), seed=0
    ).fit(X)
    assert len(model.noyaux) == 2