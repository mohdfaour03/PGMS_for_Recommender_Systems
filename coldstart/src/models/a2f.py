"""Attribute-to-factor style content model using a lightweight MLP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class A2FConfig:
    """Configuration for the attribute-to-factor mapper."""

    hidden_dim: int = 64
    lr: float = 0.01
    reg: float = 1e-4
    iters: int = 200
    batch_size: int | None = 512
    seed: int = 42


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


def _init_params(n_features: int, n_factors: int, hidden_dim: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    limit_1 = np.sqrt(6.0 / (n_features + hidden_dim))
    limit_2 = np.sqrt(6.0 / (hidden_dim + n_factors))
    params = {
        "W1": rng.uniform(-limit_1, limit_1, size=(n_features, hidden_dim)).astype(np.float32),
        "b1": np.zeros(hidden_dim, dtype=np.float32),
        "W2": rng.uniform(-limit_2, limit_2, size=(hidden_dim, n_factors)).astype(np.float32),
        "b2": np.zeros(n_factors, dtype=np.float32),
    }
    return params


def _forward(X: np.ndarray, params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z1 = X @ params["W1"] + params["b1"]
    h1 = _relu(z1)
    y_hat = h1 @ params["W2"] + params["b2"]
    return z1, h1, y_hat


def train_a2f_mlp(
    X_warm: List[List[float]],
    V_warm: List[List[float]],
    config: A2FConfig,
) -> Dict[str, np.ndarray]:
    """Train a shallow MLP to map content features into latent factors."""

    X = np.asarray(X_warm, dtype=np.float32)
    Y = np.asarray(V_warm, dtype=np.float32)
    n_samples, n_features = X.shape
    n_factors = Y.shape[1]

    rng = np.random.default_rng(config.seed)
    params = _init_params(n_features, n_factors, config.hidden_dim, rng)

    batch_size = config.batch_size or n_samples

    for _ in range(config.iters):
        indices = np.arange(n_samples)
        if batch_size < n_samples:
            rng.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            Xb = X[batch_idx]
            Yb = Y[batch_idx]

            z1, h1, y_hat = _forward(Xb, params)
            diff = (y_hat - Yb) / Xb.shape[0]

            grad_W2 = h1.T @ diff + config.reg * params["W2"]
            grad_b2 = diff.sum(axis=0)

            grad_h1 = diff @ params["W2"].T
            grad_z1 = grad_h1 * _relu_grad(z1)
            grad_W1 = Xb.T @ grad_z1 + config.reg * params["W1"]
            grad_b1 = grad_z1.sum(axis=0)

            params["W2"] -= config.lr * grad_W2
            params["b2"] -= config.lr * grad_b2
            params["W1"] -= config.lr * grad_W1
            params["b1"] -= config.lr * grad_b1

    return params


def infer_item_factors(X_cold: List[List[float]], params: Dict[str, np.ndarray]) -> List[List[float]]:
    """Project cold-item features through the trained mapper."""
    X = np.asarray(X_cold, dtype=np.float32)
    _, h1, y_hat = _forward(X, params)
    return y_hat.astype(np.float32).tolist()
