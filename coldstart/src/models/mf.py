"""Simple matrix factorisation baseline for warm interactions."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class MFConfig:
    factors: int = 32
    reg: float = 0.02
    iters: int = 20
    lr: float = 0.01
    seed: int = 42


def build_index(values: Iterable[str]) -> Tuple[Dict[str, int], List[str]]:
    unique = sorted(set(values))
    mapping = {val: idx for idx, val in enumerate(unique)}
    return mapping, unique


def build_interaction_maps(
    interactions: Sequence[dict],
    user_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int],
) -> Tuple[List[List[Tuple[int, float]]], List[List[Tuple[int, float]]]]:
    user_items: List[List[Tuple[int, float]]] = [list() for _ in user_to_idx]
    item_users: List[List[Tuple[int, float]]] = [list() for _ in item_to_idx]
    for row in interactions:
        u = user_to_idx[row["user_id"]]
        i = item_to_idx[row["item_id"]]
        r = float(row["rating_or_y"])
        user_items[u].append((i, r))
        item_users[i].append((u, r))
    return user_items, item_users


def _dot(vec_a: List[float], vec_b: List[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


def als_sgd(
    interactions: Sequence[dict],
    factors: int,
    reg: float,
    iters: int,
    lr: float,
    seed: int,
) -> Tuple[List[List[float]], List[List[float]], Dict[str, int], Dict[str, int]]:
    user_to_idx, user_ids = build_index(row["user_id"] for row in interactions)
    item_to_idx, item_ids = build_index(row["item_id"] for row in interactions)
    rng = random.Random(seed)
    U = [[rng.uniform(-0.1, 0.1) for _ in range(factors)] for _ in user_ids]
    V = [[rng.uniform(-0.1, 0.1) for _ in range(factors)] for _ in item_ids]

    for _ in range(iters):
        for row in interactions:
            u_idx = user_to_idx[row["user_id"]]
            i_idx = item_to_idx[row["item_id"]]
            rating = float(row["rating_or_y"])
            pred = _dot(U[u_idx], V[i_idx])
            err = rating - pred
            for f in range(factors):
                u_val = U[u_idx][f]
                i_val = V[i_idx][f]
                U[u_idx][f] += lr * (err * i_val - reg * u_val)
                V[i_idx][f] += lr * (err * u_val - reg * i_val)
    return U, V, user_to_idx, item_to_idx


def train_mf(
    interactions: Sequence[dict],
    factors: int = 32,
    reg: float = 0.02,
    iters: int = 20,
    lr: float = 0.01,
    seed: int = 42,
) -> Tuple[List[List[float]], List[List[float]], Dict[str, int], Dict[str, int]]:
    return als_sgd(interactions, factors, reg, iters, lr, seed)


def save_factors(U: List[List[float]], V: List[List[float]], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, matrix in [("U", U), ("V_warm", V)]:
        with (out_path / f"{name}.json").open("w", encoding="utf-8") as fh:
            for row in matrix:
                fh.write(",".join(f"{value:.6f}" for value in row) + "\n")
