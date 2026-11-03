"""Synthetic models of the paper: N1-N6 and A1-A9 (RP-CIT), and the RP-GCM diagnostic models."""
from __future__ import annotations

import math

import numpy as np

NULLS = ("N1", "N2", "N3", "N4", "N5", "N6")
ALTERNATIVES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9")
MODELS = NULLS + ALTERNATIVES
GCM_MODELS = ("N1", "N2", "A1", "A2", "A7")
Z_DISTRIBUTIONS = ("gaussian", "bounded")


def _z(rng, n, z_dist):
    if z_dist == "gaussian":
        return rng.normal(size=(n, 1))
    if z_dist == "bounded":
        return rng.uniform(-math.sqrt(3), math.sqrt(3), size=(n, 1))
    raise ValueError(f"z_dist must be one of {Z_DISTRIBUTIONS}.")


def _unit(rng, d):
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def _generic_null(model, rng, n, d, z):
    if model == "N4":
        ex, ey = rng.standard_t(5.0, size=(n, d)), rng.standard_t(5.0, size=(n, d))
    else:
        ex, ey = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    if model == "N2":
        ex = ex**2 - 1.0
    elif model == "N3":
        ex, ey = np.log1p(np.abs(ex)), np.exp(np.clip(ey, -2.0, 2.0)) - 1.0
    elif model == "N4":
        ex, ey = np.sin(ex), ey**2 - 1.0
    return z + ex, z + ey


def _couple(model, rng, x, y, z, theta, gamma):
    """A1, A2, A3 and A7: add dependence between base data X and Y."""
    n, d = x.shape
    if model == "A1":
        h = rng.normal(size=(n, 1))
        u, v = _unit(rng, d), _unit(rng, d)
        return x + gamma * h * u, y + theta * h * v
    if model == "A2":
        m = rng.normal(size=(d, d)) / math.sqrt(d)
        return x, y + theta * x @ m.T
    u, v = _unit(rng, d), _unit(rng, d)
    if model == "A3":
        return x, y + theta * np.tanh(x @ u)[:, None] * v
    gate = 1.0 / (1.0 + np.exp(-z @ rng.normal(size=z.shape[1])))
    return x, y + theta * (gate * (x @ u))[:, None] * v


def _alternative(model, rng, n, d, z, theta=2.0, gamma=2.0):
    bx, by = rng.normal(scale=0.5, size=(d, 1)), rng.normal(scale=0.5, size=(d, 1))
    x = z @ bx.T + 0.5 * rng.normal(size=(n, d))
    y = z @ by.T + 0.5 * rng.normal(size=(n, d))
    if model in ("A1", "A2", "A3", "A7"):
        return _couple(model, rng, x, y, z, theta, gamma)
    if model == "A4":
        s = rng.choice(d, size=min(5, d), replace=False)
        y[:, 0] += theta * x[:, s].sum(axis=1) / math.sqrt(len(s))
    elif model == "A5":
        y[:, 0] += theta * x[:, 0] * x[:, min(1, d - 1)]
    elif model == "A6":
        y[:, d // 2] += theta * x[:, d // 2]
    return x, y


def _localized(model, rng, n, d, z, theta=1.5):
    """A8/N5: f(t) = (t^2 - 1)/sqrt(2); A9/N6: standardized |t|."""
    s = math.ceil(math.sqrt(d))
    bx, by = rng.normal(0.0, 0.5, size=(2, d))
    scores, weights = rng.random(size=(2, d)), rng.normal(size=(2, d))
    loadings = []
    for score, weight in zip(scores, weights):
        support = np.argsort(score, kind="stable")[:s]
        v = np.zeros(d)
        v[support] = weight[support]
        loadings.append(v / np.linalg.norm(v))
    u, v = loadings
    shared, independent = rng.normal(size=(2, n))
    ex, ey = rng.normal(size=(2, n, d))
    t = shared if model in ("A8", "A9") else independent
    if model in ("A8", "N5"):
        f = (t**2 - 1.0) / math.sqrt(2.0)
    else:
        f = (np.abs(t) - math.sqrt(2 / math.pi)) / math.sqrt(1 - 2 / math.pi)
    kappa = theta / math.sqrt(1.0 + theta**2)
    noise = 1.0 / math.sqrt(1.0 + theta**2)
    x = z * bx + 0.5 * (kappa * math.sqrt(d) * shared[:, None] * u + noise * ex)
    y = z * by + 0.5 * (kappa * math.sqrt(d) * f[:, None] * v + noise * ey)
    return x, y


def generate(model, n, d, rng, z_dist="gaussian"):
    """Return X (n, d), Y (n, d) and scalar Z (n, 1) for one replication."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}.")
    z = _z(rng, n, z_dist)
    if model in ("N1", "N2", "N3", "N4"):
        x, y = _generic_null(model, rng, n, d, z)
    elif model in ("N5", "N6", "A8", "A9"):
        x, y = _localized(model, rng, n, d, z)
    else:
        x, y = _alternative(model, rng, n, d, z)
    return x, y, z


def generate_gcm(model, n, dz, rng, d=10):
    """RP-GCM diagnostic models: Z ~ N(0, I_dz), conditional means b(Z) = first d
    coordinates of Z, repeated cyclically when dz < d."""
    if model not in GCM_MODELS:
        raise ValueError(f"model must be one of {GCM_MODELS}.")
    z = rng.normal(size=(n, dz))
    ex, ey = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    b = np.tile(z, (1, math.ceil(d / dz)))[:, :d]
    x, y = b + (ex**2 - 1.0 if model == "N2" else ex), b + ey
    if model in ("A1", "A2", "A7"):
        theta = {"A1": 0.5, "A2": 0.3, "A7": 0.8}[model]
        x, y = _couple(model, rng, x, y, z, theta, gamma=0.5)
    return x, y, z
