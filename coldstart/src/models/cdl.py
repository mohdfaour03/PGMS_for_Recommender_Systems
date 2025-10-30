"""Simplified Collaborative Deep Learning (CDL) style model.

The original CDL couples a stacked denoising autoencoder with collaborative
matrix factorisation. This implementation keeps the spirit of the approach by
training a single-layer denoising autoencoder on the item text features and
learning a ridge-regression mapping from the hidden representation into the
latent factor space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class CDLConfig:
    hidden_dim: int = 128
    lr: float = 0.01
    reg: float = 1e-4
    map_reg: float = 1e-2
    iters: int = 200
    batch_size: int | None = 256
    corruption: float = 0.1
    seed: int = 42


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


def _init_params(n_features: int, hidden_dim: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    limit = np.sqrt(6.0 / (n_features + hidden_dim))
    params = {
        "W1": rng.uniform(-limit, limit, size=(n_features, hidden_dim)).astype(np.float32),
        "b1": np.zeros(hidden_dim, dtype=np.float32),
        "W2": rng.uniform(-limit, limit, size=(hidden_dim, n_features)).astype(np.float32),
        "b2": np.zeros(n_features, dtype=np.float32),
    }
    return params


def _encode(X: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    return _relu(X @ params["W1"] + params["b1"])


def train_cdl(
    warm_features: List[List[float]],
    warm_factors: List[List[float]],
    config: CDLConfig,
) -> Dict[str, np.ndarray]:
    X = np.asarray(warm_features, dtype=np.float32)
    V = np.asarray(warm_factors, dtype=np.float32)
    if X.size == 0 or V.size == 0:
        raise ValueError("CDL requires non-empty warm features and factors.")

    n_samples, n_features = X.shape

    rng = np.random.default_rng(config.seed)
    params = _init_params(n_features, config.hidden_dim, rng)
    batch_size = config.batch_size or n_samples

    for _ in range(config.iters):
        indices = np.arange(n_samples)
        if batch_size < n_samples:
            rng.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            idx = indices[start : start + batch_size]
            Xb = X[idx]
            mask = (rng.random(Xb.shape) >= config.corruption).astype(np.float32)
            corrupted = Xb * mask

            z = corrupted @ params["W1"] + params["b1"]
            h = _relu(z)
            recon = h @ params["W2"] + params["b2"]

            diff = (recon - Xb) / Xb.shape[0]

            grad_W2 = h.T @ diff + config.reg * params["W2"]
            grad_b2 = diff.sum(axis=0)

            grad_h = diff @ params["W2"].T
            grad_z = grad_h * _relu_grad(z)
            grad_W1 = corrupted.T @ grad_z + config.reg * params["W1"]
            grad_b1 = grad_z.sum(axis=0)

            params["W2"] -= config.lr * grad_W2
            params["b2"] -= config.lr * grad_b2
            params["W1"] -= config.lr * grad_W1
            params["b1"] -= config.lr * grad_b1

    hidden = _encode(X, params)
    HtH = hidden.T @ hidden + config.map_reg * np.eye(config.hidden_dim, dtype=np.float32)
    hidden_to_factor = np.linalg.solve(HtH, hidden.T @ V).astype(np.float32)

    params["hidden_to_factor"] = hidden_to_factor
    return params


def infer_item_factors(
    features: List[List[float]],
    params: Dict[str, np.ndarray],
) -> List[List[float]]:
    if not features:
        return []
    X = np.asarray(features, dtype=np.float32)
    hidden = _encode(X, params)
    factors = hidden @ params["hidden_to_factor"]
    return factors.astype(np.float32).tolist()


__all__ = ["CDLConfig", "train_cdl", "infer_item_factors"]
