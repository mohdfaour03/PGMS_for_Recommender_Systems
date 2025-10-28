"""End-to-end routines for the cold-start benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

from . import data_io
from .evaluation import hit_ndcg_at_k
from .models import ctr_lite, mf
from .split_strict import persist_split, strict_cold_split
from .text_featurizer import TfidfTextFeaturizer


def _unique_item_text(interactions: Sequence[dict]) -> Dict[str, str]:
    item_text: Dict[str, str] = {}
    for row in interactions:
        item_text.setdefault(row["item_id"], row["item_text"])
    return item_text


def prepare_dataset(
    data_path: str | Path,
    out_dir: str | Path,
    tfidf_params: dict,
    cold_item_frac: float = 0.15,
    seed: int = 42,
) -> None:
    interactions = data_io.load_interactions(data_path)
    warm_rows, cold_rows, cold_items = strict_cold_split(
        interactions, cold_item_frac=cold_item_frac, seed=seed
    )
    persist_split(warm_rows, cold_rows, cold_items, out_dir)

    warm_item_text = _unique_item_text(warm_rows)
    cold_item_text = _unique_item_text(cold_rows)
    warm_item_ids = sorted(warm_item_text)
    cold_item_ids = sorted(cold_item_text)

    featurizer = TfidfTextFeaturizer(**tfidf_params)
    warm_texts = [warm_item_text[item] for item in warm_item_ids]
    warm_features = featurizer.fit_transform_warm(warm_texts)
    cold_texts = [cold_item_text[item] for item in cold_item_ids]
    cold_features = featurizer.transform(cold_texts)
    print(f"Warm text features shape: ({len(warm_features)}, {len(warm_features[0]) if warm_features else 0})")
    print(f"Cold text features shape: ({len(cold_features)}, {len(cold_features[0]) if cold_features else 0})")

    out_path = Path(out_dir)
    data_io.save_json(warm_item_ids, out_path / "warm_item_ids.json")
    data_io.save_json(cold_item_ids, out_path / "cold_item_ids.json")
    data_io.save_matrix(warm_features, out_path / "warm_item_text_features.json")
    data_io.save_matrix(cold_features, out_path / "cold_item_text_features.json")
    data_io.save_json(featurizer.save_state(), out_path / "tfidf_state.json")


def _refit_users(
    user_items: Sequence[Sequence[tuple[int, float]]],
    item_factors: Sequence[Sequence[float]],
    factors: int,
    reg: float,
    lr: float,
    iters: int,
) -> list[list[float]]:
    U = [[0.0 for _ in range(factors)] for _ in user_items]
    for _ in range(iters):
        for u_idx, interactions in enumerate(user_items):
            if not interactions:
                continue
            for item_idx, rating in interactions:
                pred = sum(U[u_idx][f] * item_factors[item_idx][f] for f in range(factors))
                err = rating - pred
                for f in range(factors):
                    u_val = U[u_idx][f]
                    i_val = item_factors[item_idx][f]
                    U[u_idx][f] += lr * (err * i_val - reg * u_val)
    return U


def train_and_evaluate_ctrlite(
    data_dir: str | Path,
    k_factors: int = 32,
    k_eval: int = 10,
    mf_reg: float = 0.02,
    mf_iters: int = 30,
    mf_lr: float = 0.02,
    seed: int = 42,
    ctrlite_reg: float = 0.01,
    ctrlite_lr: float = 0.1,
    ctrlite_iters: int = 80,
    adaptive: bool = False,
) -> Dict[str, Dict[str, float]]:
    data_path = Path(data_dir)
    warm_rows = data_io.load_interactions(data_path / "warm_interactions.csv")
    cold_rows = data_io.load_interactions(data_path / "cold_interactions.csv")
    warm_item_ids = data_io.load_json(data_path / "warm_item_ids.json")
    cold_item_ids = data_io.load_json(data_path / "cold_item_ids.json")
    warm_features = data_io.load_matrix(data_path / "warm_item_text_features.json")
    cold_features = data_io.load_matrix(data_path / "cold_item_text_features.json")

    U, V, user_to_idx, item_to_idx = mf.train_mf(
        warm_rows,
        factors=k_factors,
        reg=mf_reg,
        iters=mf_iters,
        lr=mf_lr,
        seed=seed,
    )
    models_dir = data_path / "models"
    mf.save_factors(U, V, models_dir)

    order = [item_to_idx[item] for item in warm_item_ids]
    V_ordered = [V[idx] for idx in order]

    ctrl_cfg = ctr_lite.CtrliteConfig(reg=ctrlite_reg, lr=ctrlite_lr, iters=ctrlite_iters)
    W = ctr_lite.train_text_to_factors(warm_features, V_ordered, ctrl_cfg)
    V_cold = ctr_lite.infer_cold_item_factors(cold_features, W)
    scores = ctr_lite.score_users_on_cold(U, V_cold)
    base_metrics = hit_ndcg_at_k(scores, user_to_idx, cold_item_ids, cold_rows, k=k_eval)

    results: Dict[str, Dict[str, float]] = {"ctrlite": base_metrics}

    if adaptive:
        warm_counts: Dict[str, int] = {}
        for row in warm_rows:
            warm_counts[row["item_id"]] = warm_counts.get(row["item_id"], 0) + 1
        max_count = max(warm_counts.values()) if warm_counts else 1
        rarity = [1.0 - (warm_counts.get(item, 0) / max_count) for item in warm_item_ids]
        warm_pred = ctr_lite.infer_cold_item_factors(warm_features, W)
        blend = []
        for idx, base_vec in enumerate(V_ordered):
            blended = []
            weight = rarity[idx]
            for f in range(len(base_vec)):
                blended.append(weight * warm_pred[idx][f] + (1.0 - weight) * base_vec[f])
            blend.append(blended)
        user_items, _ = mf.build_interaction_maps(warm_rows, user_to_idx, item_to_idx)
        U_adapt = _refit_users(user_items, blend, k_factors, mf_reg, mf_lr, max(5, mf_iters // 2))
        scores_adapt = ctr_lite.score_users_on_cold(U_adapt, V_cold)
        adaptive_metrics = hit_ndcg_at_k(
            scores_adapt, user_to_idx, cold_item_ids, cold_rows, k=k_eval
        )
        results["ctrlite_adaptive"] = adaptive_metrics

    return results
