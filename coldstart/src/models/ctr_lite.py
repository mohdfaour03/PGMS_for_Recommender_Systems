"""Text-to-factor mapping using a lightweight gradient-descent model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class CtrliteConfig:
    reg: float = 0.01
    lr: float = 0.1
    iters: int = 50


def train_text_to_factors(
    X_warm: List[List[float]], V_warm: List[List[float]], config: CtrliteConfig
) -> List[List[float]]:
    if not X_warm:
        return []
    n_features = len(X_warm[0])
    n_factors = len(V_warm[0]) if V_warm else 0
    W = [[0.0 for _ in range(n_factors)] for _ in range(n_features)]

    for _ in range(config.iters):
        for x_vec, target in zip(X_warm, V_warm):
            pred = [sum(W[f][k] * x_vec[f] for f in range(n_features)) for k in range(n_factors)]
            for k in range(n_factors):
                error = pred[k] - target[k]
                for f in range(n_features):
                    x_val = x_vec[f]
                    if x_val == 0.0:
                        continue
                    W[f][k] -= config.lr * (error * x_val + config.reg * W[f][k])
    return W


def infer_cold_item_factors(X_cold: List[List[float]], W: List[List[float]]) -> List[List[float]]:
    n_features = len(W)
    n_factors = len(W[0]) if W else 0
    factors: List[List[float]] = []
    for x_vec in X_cold:
        row = [0.0] * n_factors
        for f in range(n_features):
            x_val = x_vec[f]
            if x_val == 0.0:
                continue
            for k in range(n_factors):
                row[k] += x_val * W[f][k]
        factors.append(row)
    return factors


def score_users_on_cold(U: List[List[float]], V_cold: List[List[float]]) -> List[List[float]]:
    scores: List[List[float]] = []
    for u_vec in U:
        row_scores = []
        for item_vec in V_cold:
            row_scores.append(_dot(u_vec, item_vec))
        scores.append(row_scores)
    return scores
