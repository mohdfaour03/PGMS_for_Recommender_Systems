"""Simplified Collaborative Topic Poisson Factorization implementation.

This module captures the core intuition of CTPF by learning low-rank topic
mixtures for warm items and linking them to collaborative filtering factors.
It omits the full Poisson likelihood in favour of multiplicative-update NMF
followed by a ridge-regression mapping into the latent space, which keeps the
implementation lightweight and dependency free while remaining useful for
notebook comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CTPFConfig:
    topics: int = 50
    nmf_iters: int = 200
    reg: float = 0.1
    projection_iters: int = 100
    seed: int = 42


def _nmf(
    X: np.ndarray,
    topics: int,
    iters: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_items, n_features = X.shape
    W = rng.random((n_items, topics), dtype=np.float32) + 1e-3
    H = rng.random((topics, n_features), dtype=np.float32) + 1e-3
    eps = 1e-8

    for _ in range(iters):
        WH = W @ H + eps
        W *= (X / WH) @ H.T
        W = np.maximum(W, eps)
        W /= np.maximum(W.sum(axis=1, keepdims=True), eps)

        WH = W @ H + eps
        H *= W.T @ (X / WH)
        H = np.maximum(H, eps)

    return W, H


def _project_topics(X: np.ndarray, H: np.ndarray, iters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_items, _ = X.shape
    topics = H.shape[0]
    W = rng.random((n_items, topics), dtype=np.float32) + 1e-3
    eps = 1e-8

    HT = H.T
    for _ in range(iters):
        WH = W @ H
        numerator = X @ HT
        denominator = WH @ HT + eps
        W *= numerator / np.maximum(denominator, eps)
        W = np.maximum(W, eps)
        W /= np.maximum(W.sum(axis=1, keepdims=True), eps)
    return W


def train_ctpf(
    warm_features: List[List[float]],
    warm_factors: List[List[float]],
    config: CTPFConfig,
) -> Dict[str, np.ndarray]:
    X = np.asarray(warm_features, dtype=np.float32)
    V = np.asarray(warm_factors, dtype=np.float32)
    if X.size == 0 or V.size == 0:
        raise ValueError("CTPF requires non-empty warm features and factors.")

    W_topics, H_topics = _nmf(X, config.topics, config.nmf_iters, config.seed)

    XtX = W_topics.T @ W_topics + config.reg * np.eye(config.topics, dtype=np.float32)
    topic_to_factor = np.linalg.solve(XtX, W_topics.T @ V).astype(np.float32)

    return {
        "topic_components": H_topics.astype(np.float32),
        "topic_to_factor": topic_to_factor,
    }


def infer_item_factors(
    features: List[List[float]],
    params: Dict[str, np.ndarray],
    projection_iters: int,
    seed: int,
) -> List[List[float]]:
    if not features:
        return []
    X = np.asarray(features, dtype=np.float32)
    H = params["topic_components"]
    topics = H.shape[0]
    W = _project_topics(X, H, projection_iters, seed)
    topic_to_factor = params["topic_to_factor"]
    return (W @ topic_to_factor).astype(np.float32).tolist()


__all__ = ["CTPFConfig", "train_ctpf", "infer_item_factors"]
