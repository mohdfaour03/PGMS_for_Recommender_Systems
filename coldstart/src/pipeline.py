"""End-to-end routines for the cold-start benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

from . import data_io
from .evaluation import hit_ndcg_at_k
from .models import a2f, cdl, ctr_lite, ctpf, hft, mf
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
    interaction_limit: int | None = None,
) -> None:
    interactions = data_io.load_interactions(data_path, limit=interaction_limit)
    if interaction_limit is not None:
        print(f"Using {len(interactions)} interactions out of the source data (limit={interaction_limit}).")
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


def train_and_evaluate_content_model(
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
    model: str = "ctrlite",
    a2f_cfg: dict | None = None,
    ctpf_cfg: dict | None = None,
    cdl_cfg: dict | None = None,
    hft_cfg: dict | None = None,
    mf_cfg: dict | None = None,
    backend: str = "numpy",
    prefer_gpu: bool = True,
) -> Dict[str, Dict[str, float]]:
    data_path = Path(data_dir)
    warm_rows = data_io.load_interactions(data_path / "warm_interactions.csv")
    cold_rows = data_io.load_interactions(data_path / "cold_interactions.csv")
    warm_item_ids = data_io.load_json(data_path / "warm_item_ids.json")
    cold_item_ids = data_io.load_json(data_path / "cold_item_ids.json")
    warm_features = data_io.load_matrix(data_path / "warm_item_text_features.json")
    cold_features = data_io.load_matrix(data_path / "cold_item_text_features.json")

    backend_key = backend.lower()
    use_torch = backend_key == "torch"
    mf_cfg = dict(mf_cfg or {})
    if use_torch:
        from . import torch_backend

        batch_size = int(mf_cfg.get("batch_size", 4096))
        U, V, user_to_idx, item_to_idx = torch_backend.train_mf(
            warm_rows,
            factors=k_factors,
            reg=mf_reg,
            iters=mf_iters,
            lr=mf_lr,
            seed=seed,
            batch_size=batch_size,
            prefer_gpu=prefer_gpu,
        )
    else:
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

    available_models = ("ctrlite", "a2f", "ctpf", "cdl", "hft")
    requested: list[str]
    if isinstance(model, str):
        model_lower = model.lower()
        if model_lower == "all":
            requested = list(available_models)
        else:
            requested = [part.strip().lower() for part in model.split(",") if part.strip()]
    else:
        requested = [str(part).lower() for part in model]
        if "all" in requested:
            requested = list(available_models)
    requested = list(dict.fromkeys(requested))
    if not requested:
        requested = ["ctrlite"]

    invalid = [name for name in requested if name not in available_models]
    if invalid:
        raise ValueError(f"Unknown model(s) requested: {invalid}")

    results: Dict[str, Dict[str, float]] = {}
    ctrlite_cache: Dict[str, object] = {}
    infer_batch_size = max(1, int(mf_cfg.get("infer_batch_size", 4096)))
    score_batch_size = max(1, int(mf_cfg.get("score_batch_size", 4096)))

    def _score(users: list[list[float]], items: list[list[float]]) -> list[list[float]]:
        if use_torch:
            return torch_backend.score_users_on_cold(
                users, items, batch_size=score_batch_size, prefer_gpu=prefer_gpu
            )
        return ctr_lite.score_users_on_cold(users, items)

    for name in requested:
        if name == "ctrlite":
            if use_torch:
                ctrlite_batch = max(1, int(mf_cfg.get("ctrlite_batch_size", 2048)))
                ctrl_config = torch_backend.CtrliteTorchConfig(
                    reg=float(ctrlite_reg),
                    lr=float(ctrlite_lr),
                    iters=int(ctrlite_iters),
                    batch_size=ctrlite_batch,
                )
                ctrlite_model = torch_backend.train_ctrlite(
                    warm_features, V_ordered, ctrl_config, prefer_gpu=prefer_gpu
                )
                V_cold = torch_backend.infer_ctrlite(
                    ctrlite_model,
                    cold_features,
                    prefer_gpu=prefer_gpu,
                    batch_size=infer_batch_size,
                )
                scores = _score(U, V_cold)
                ctrlite_cache["model"] = ctrlite_model
            else:
                ctrl_config = ctr_lite.CtrliteConfig(
                    reg=ctrlite_reg, lr=ctrlite_lr, iters=ctrlite_iters
                )
                W = ctr_lite.train_text_to_factors(warm_features, V_ordered, ctrl_config)
                V_cold = ctr_lite.infer_cold_item_factors(cold_features, W)
                scores = _score(U, V_cold)
                ctrlite_cache["W"] = W
            ctrlite_cache["V_cold"] = V_cold
            metrics = hit_ndcg_at_k(scores, user_to_idx, cold_item_ids, cold_rows, k=k_eval)
            results["ctrlite"] = metrics
        elif name == "a2f":
            cfg = a2f_cfg or {}
            hidden_dim = int(cfg.get("hidden_dim", 128))
            lr_a2f = float(cfg.get("lr", 0.01))
            reg_a2f = float(cfg.get("reg", 1e-4))
            iters_a2f = int(cfg.get("iters", 200))
            batch_cfg = cfg.get("batch_size", 512)
            batch_a2f = None if batch_cfg in (None, "None") else int(batch_cfg)
            seed_a2f = int(cfg.get("seed", seed))
            if use_torch:
                torch_cfg = torch_backend.A2FTorchConfig(
                    hidden_dim=hidden_dim,
                    lr=lr_a2f,
                    reg=reg_a2f,
                    iters=iters_a2f,
                    batch_size=batch_a2f or 512,
                )
                a2f_model = torch_backend.train_a2f(
                    warm_features, V_ordered, torch_cfg, prefer_gpu=prefer_gpu
                )
                V_cold = torch_backend.infer_a2f(
                    a2f_model,
                    cold_features,
                    prefer_gpu=prefer_gpu,
                    batch_size=infer_batch_size,
                )
            else:
                cpu_cfg = a2f.A2FConfig(
                    hidden_dim=hidden_dim,
                    lr=lr_a2f,
                    reg=reg_a2f,
                    iters=iters_a2f,
                    batch_size=batch_a2f,
                    seed=seed_a2f,
                )
                params = a2f.train_a2f_mlp(warm_features, V_ordered, cpu_cfg)
                V_cold = a2f.infer_item_factors(cold_features, params)
            scores = _score(U, V_cold)
            metrics = hit_ndcg_at_k(scores, user_to_idx, cold_item_ids, cold_rows, k=k_eval)
            results["a2f"] = metrics
        elif name == "ctpf":
            cfg = ctpf_cfg or {}
            topics = int(cfg.get("topics", 80))
            nmf_iters = int(cfg.get("nmf_iters", 200))
            reg_ctpf = float(cfg.get("reg", 0.1))
            projection_iters = int(cfg.get("projection_iters", 120))
            seed_ctpf = int(cfg.get("seed", seed))
            if use_torch:
                params = torch_backend.train_ctpf(
                    warm_features,
                    V_ordered,
                    topics=topics,
                    nmf_iters=nmf_iters,
                    reg=reg_ctpf,
                    prefer_gpu=prefer_gpu,
                    seed=seed_ctpf,
                )
                V_cold = torch_backend.infer_ctpf(
                    cold_features,
                    params,
                    projection_iters,
                    seed_ctpf + 1,
                    prefer_gpu=prefer_gpu,
                )
            else:
                cpu_cfg = ctpf.CTPFConfig(
                    topics=topics,
                    nmf_iters=nmf_iters,
                    reg=reg_ctpf,
                    projection_iters=projection_iters,
                    seed=seed_ctpf,
                )
                params = ctpf.train_ctpf(warm_features, V_ordered, cpu_cfg)
                V_cold = ctpf.infer_item_factors(
                    cold_features, params, projection_iters, seed_ctpf + 1
                )
            scores = _score(U, V_cold)
            metrics = hit_ndcg_at_k(scores, user_to_idx, cold_item_ids, cold_rows, k=k_eval)
            results["ctpf"] = metrics
        elif name == "cdl":
            cfg = cdl_cfg or {}
            hidden_dim = int(cfg.get("hidden_dim", 256))
            lr_cdl = float(cfg.get("lr", 5e-3))
            reg_cdl = float(cfg.get("reg", 1e-4))
            iters_cdl = int(cfg.get("iters", 300))
            batch_cfg = cfg.get("batch_size", 256)
            batch_cdl = None if batch_cfg in (None, "None") else int(batch_cfg)
            corruption = float(cfg.get("corruption", 0.2))
            factor_weight = float(cfg.get("factor_weight", 1.0))
            map_reg = float(cfg.get("map_reg", 1e-2))
            seed_cdl = int(cfg.get("seed", seed))
            if use_torch:
                torch_cfg = torch_backend.CDLTorchConfig(
                    hidden_dim=hidden_dim,
                    lr=lr_cdl,
                    reg=reg_cdl,
                    iters=iters_cdl,
                    batch_size=batch_cdl or 256,
                    corruption=corruption,
                    factor_weight=factor_weight,
                )
                cdl_model = torch_backend.train_cdl(
                    warm_features, V_ordered, torch_cfg, prefer_gpu=prefer_gpu
                )
                V_cold = torch_backend.infer_cdl(
                    cdl_model,
                    cold_features,
                    prefer_gpu=prefer_gpu,
                    batch_size=infer_batch_size,
                )
            else:
                cpu_cfg = cdl.CDLConfig(
                    hidden_dim=hidden_dim,
                    lr=lr_cdl,
                    reg=reg_cdl,
                    map_reg=map_reg,
                    iters=iters_cdl,
                    batch_size=batch_cdl,
                    corruption=corruption,
                    seed=seed_cdl,
                )
                params = cdl.train_cdl(warm_features, V_ordered, cpu_cfg)
                V_cold = cdl.infer_item_factors(cold_features, params)
            scores = _score(U, V_cold)
            metrics = hit_ndcg_at_k(scores, user_to_idx, cold_item_ids, cold_rows, k=k_eval)
            results["cdl"] = metrics
        elif name == "hft":
            cfg = hft_cfg or {}
            topics = int(cfg.get("topics", 50))
            nmf_iters = int(cfg.get("nmf_iters", 200))
            kappa = float(cfg.get("kappa", 1.0))
            reg_hft = float(cfg.get("reg", 0.05))
            projection_iters = int(cfg.get("projection_iters", 100))
            seed_hft = int(cfg.get("seed", seed))
            if use_torch:
                params = torch_backend.train_hft(
                    warm_features,
                    V_ordered,
                    topics=topics,
                    nmf_iters=nmf_iters,
                    kappa=kappa,
                    reg=reg_hft,
                    prefer_gpu=prefer_gpu,
                    seed=seed_hft,
                )
                V_cold = torch_backend.infer_hft(
                    cold_features,
                    params,
                    projection_iters,
                    seed_hft + 1,
                    prefer_gpu=prefer_gpu,
                )
            else:
                cpu_cfg = hft.HFTConfig(
                    topics=topics,
                    nmf_iters=nmf_iters,
                    kappa=kappa,
                    reg=reg_hft,
                    projection_iters=projection_iters,
                    seed=seed_hft,
                )
                params = hft.train_hft(warm_features, V_ordered, cpu_cfg)
                V_cold = hft.infer_item_factors(
                    cold_features, params, projection_iters, seed_hft + 1, cpu_cfg.kappa
                )
            scores = _score(U, V_cold)
            metrics = hit_ndcg_at_k(scores, user_to_idx, cold_item_ids, cold_rows, k=k_eval)
            results["hft"] = metrics

    if adaptive and "ctrlite" not in requested:
        print("Adaptive user refit is only available for ctrlite; skipping.")
        adaptive = False

    if adaptive and ctrlite_cache:
        warm_counts: Dict[str, int] = {}
        for row in warm_rows:
            warm_counts[row["item_id"]] = warm_counts.get(row["item_id"], 0) + 1
        max_count = max(warm_counts.values()) if warm_counts else 1
        rarity = [1.0 - (warm_counts.get(item, 0) / max_count) for item in warm_item_ids]
        if use_torch and "model" in ctrlite_cache:
            warm_pred = torch_backend.infer_ctrlite(
                ctrlite_cache["model"],
                warm_features,
                prefer_gpu=prefer_gpu,
                batch_size=infer_batch_size,
            )
        else:
            warm_pred = ctr_lite.infer_cold_item_factors(warm_features, ctrlite_cache["W"])
        blend = []
        for idx, base_vec in enumerate(V_ordered):
            weight = rarity[idx]
            predicted = warm_pred[idx]
            blended = [
                weight * predicted[f] + (1.0 - weight) * base_vec[f] for f in range(len(base_vec))
            ]
            blend.append(blended)
        user_items, _ = mf.build_interaction_maps(warm_rows, user_to_idx, item_to_idx)
        U_adapt = _refit_users(
            user_items, blend, k_factors, mf_reg, mf_lr, max(5, mf_iters // 2)
        )
        scores_adapt = _score(U_adapt, ctrlite_cache["V_cold"])
        adaptive_metrics = hit_ndcg_at_k(
            scores_adapt, user_to_idx, cold_item_ids, cold_rows, k=k_eval
        )
        results["ctrlite_adaptive"] = adaptive_metrics

    return results


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
    """Backwards-compatible wrapper retaining the original API."""
    return train_and_evaluate_content_model(
        data_dir,
        k_factors=k_factors,
        k_eval=k_eval,
        mf_reg=mf_reg,
        mf_iters=mf_iters,
        mf_lr=mf_lr,
        seed=seed,
        ctrlite_reg=ctrlite_reg,
        ctrlite_lr=ctrlite_lr,
        ctrlite_iters=ctrlite_iters,
        adaptive=adaptive,
        model="ctrlite",
        a2f_cfg=None,
    )
