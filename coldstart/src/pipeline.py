"""End-to-end routines for the cold-start benchmark."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence

from . import data_io
from .evaluation import hit_ndcg_at_k
from .exposure import ExposureConfig, load_exposure_checkpoint, train_exposure_model
from .models import a2f, cdl, ctr_lite, ctpf, hft, mf
from .split_strict import persist_split, strict_cold_split
from .text_featurizer import build_text_featurizer


def _unique_item_text(interactions: Sequence[dict]) -> Dict[str, str]:
    item_text: Dict[str, str] = {}
    for row in interactions:
        item_text.setdefault(row["item_id"], row["item_text"])
    return item_text


def _popularity_quantiles(values: Sequence[int], fractions: Sequence[float]) -> list[int]:
    if not values:
        return [0 for _ in fractions]
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    thresholds: list[int] = []
    for frac in fractions:
        idx = min(n - 1, max(0, int(round(frac * (n - 1)))))
        thresholds.append(sorted_vals[idx])
    return thresholds


def _assign_popularity_bins(
    freq_map: Dict[str, int],
    head_frac: float,
    mid_frac: float,
) -> Dict[str, str]:
    if not freq_map:
        return {}
    items = sorted(freq_map.items(), key=lambda kv: kv[1], reverse=True)
    n = len(items)
    head_cut = max(1, int(round(head_frac * n)))
    mid_cut = max(head_cut, int(round((head_frac + mid_frac) * n)))
    bins: Dict[str, str] = {}
    for idx, (item_id, _) in enumerate(items):
        if idx < head_cut:
            bins[item_id] = "head"
        elif idx < mid_cut:
            bins[item_id] = "mid"
        else:
            bins[item_id] = "tail"
    return bins


def _quantile_thresholds(values: Sequence[int], fractions: Sequence[float]) -> list[int]:
    if not values:
        return []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    thresholds: list[int] = []
    for frac in fractions:
        if n == 1:
            thresholds.append(sorted_vals[0])
            continue
        idx = min(n - 1, max(0, int(round(frac * (n - 1)))))
        thresholds.append(sorted_vals[idx])
    return thresholds


def _assign_bucket(value: int, thresholds: Sequence[int], labels: Sequence[str]) -> str:
    if not labels:
        return "all"
    for threshold, label in zip(thresholds, labels):
        if value <= threshold:
            return label
    return labels[-1]


def _token_count(text: str | None) -> int:
    if not text:
        return 0
    return len([token for token in text.split() if token])


def _log_bucket_text_coverage(
    tag: str,
    item_ids: Sequence[str],
    text_lookup: Dict[str, str],
    bucket_lookup: Dict[str, str],
) -> None:
    if not item_ids:
        return
    stats: Dict[str, Dict[str, int]] = {}
    for item_id in item_ids:
        bucket = bucket_lookup.get(item_id, "unknown")
        entry = stats.setdefault(bucket, {"total": 0, "non_empty": 0})
        entry["total"] += 1
        if _token_count(text_lookup.get(item_id)) > 0:
            entry["non_empty"] += 1
    for bucket, entry in sorted(stats.items()):
        total = entry["total"]
        if total <= 0:
            continue
        pct = entry["non_empty"] / total * 100.0
        print(f"[text] {tag} bucket={bucket}: non_empty={entry['non_empty']}/{total} ({pct:.2f}%)")


def _log_tokenizer_bucket_stats(featurizer: object, tag: str) -> None:
    length_getter = getattr(featurizer, "get_last_token_lengths", None)
    if not callable(length_getter):
        return
    lengths = length_getter()
    if not lengths:
        return
    bucket_getter = getattr(featurizer, "get_last_token_buckets", None)
    buckets: Sequence[str] = []
    if callable(bucket_getter):
        buckets = bucket_getter()
    if not buckets:
        buckets = ["all"] * len(lengths)
    stats: Dict[str, Dict[str, float]] = {}
    for length, bucket in zip(lengths, buckets):
        content = max(length - 2, 0)
        entry = stats.setdefault(bucket or "unknown", {"total": 0.0, "sum": 0.0, "non_empty": 0.0})
        entry["total"] += 1.0
        entry["sum"] += content
        if content > 0:
            entry["non_empty"] += 1.0
    max_tokens = getattr(featurizer, "max_length", "n/a")
    for bucket, entry in sorted(stats.items()):
        total = entry["total"]
        if total <= 0:
            continue
        mean_len = entry["sum"] / total
        pct = entry["non_empty"] / total * 100.0
        print(
            f"[text] {tag} bucket={bucket}: "
            f"mean_tokens={mean_len:.1f} non_empty_pct={pct:.2f}% max_seq_len={max_tokens}"
        )


def _build_popularity_bias(
    cold_item_ids: Sequence[str],
    freq_map: Dict[str, int],
    head_frac: float,
    mid_frac: float,
    beta_tail: float,
    gamma_pop: float,
) -> list[float]:
    if not cold_item_ids:
        return []
    freq_bins = _assign_popularity_bins(freq_map, head_frac, mid_frac)
    bias: list[float] = []
    for item_id in cold_item_ids:
        freq = freq_map.get(item_id, 0)
        term = gamma_pop * math.log1p(freq)
        if freq_bins.get(item_id, "tail") == "tail":
            term += max(0.0, beta_tail)
        bias.append(term)
    return bias


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return default
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered or lowered in {"none", "null"}:
            return default
    return int(value)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered or lowered in {"none", "null"}:
            return None
        return float(value)
    return float(value)


def prepare_dataset(
    data_path: str | Path,
    out_dir: str | Path,
    tfidf_params: dict | None = None,
    encoder_type: str = "tfidf",
    encoder_params: dict | None = None,
    cold_item_frac: float = 0.15,
    val_item_frac: float | None = None,
    seed: int = 42,
    interaction_limit: int | None = None,
) -> None:
    interactions = data_io.load_interactions(data_path, limit=interaction_limit)
    if interaction_limit is not None:
        print(f"Using {len(interactions)} interactions out of the source data (limit={interaction_limit}).")
    warm_rows, val_rows, cold_rows, val_items, cold_items = strict_cold_split(
        interactions, cold_item_frac=cold_item_frac, val_item_frac=val_item_frac, seed=seed
    )
    persist_split(
        warm_rows,
        cold_rows,
        cold_items,
        out_dir,
        val_rows=val_rows if val_rows else None,
        val_items=val_items if val_items else None,
    )

    warm_item_text = _unique_item_text(warm_rows)
    val_item_text = _unique_item_text(val_rows)
    cold_item_text = _unique_item_text(cold_rows)
    warm_item_ids = sorted(warm_item_text)
    val_item_ids = sorted(val_item_text)
    cold_item_ids = sorted(cold_item_text)

    text_len_lookup: Dict[str, int] = {}
    for source in (warm_rows, val_rows, cold_rows):
        for row in source:
            text_len = row.get("text_len")
            if text_len is None:
                continue
            item_id = row["item_id"]
            if item_id not in text_len_lookup:
                text_len_lookup[item_id] = int(text_len)
    text_thresholds = _quantile_thresholds(list(text_len_lookup.values()), (0.33, 0.66))
    text_labels = ["short", "medium", "long"]
    all_item_ids = warm_item_ids + val_item_ids + cold_item_ids
    text_bucket_lookup = {
        item_id: _assign_bucket(text_len_lookup.get(item_id, 0), text_thresholds, text_labels)
        for item_id in all_item_ids
    }
    _log_bucket_text_coverage("warm_text", warm_item_ids, warm_item_text, text_bucket_lookup)
    _log_bucket_text_coverage("cold_text", cold_item_ids, cold_item_text, text_bucket_lookup)

    encoder_kwargs = dict(encoder_params or tfidf_params or {})
    featurizer = build_text_featurizer(encoder_type, encoder_kwargs)
    warm_texts = [warm_item_text[item] for item in warm_item_ids]
    warm_bucket_ids = [text_bucket_lookup.get(item, "unknown") for item in warm_item_ids]
    warm_features = featurizer.fit_transform_warm(
        warm_texts,
        bucket_ids=warm_bucket_ids,
        log_prefix="[text][warm]",
    )
    _log_tokenizer_bucket_stats(featurizer, "warm_tokenized")
    if val_item_ids:
        val_bucket_ids = [text_bucket_lookup.get(item, "unknown") for item in val_item_ids]
        val_texts = [val_item_text[item] for item in val_item_ids]
        val_features = featurizer.transform(
            val_texts,
            bucket_ids=val_bucket_ids,
            log_prefix="[text][val]",
        )
        _log_tokenizer_bucket_stats(featurizer, "val_tokenized")
    else:
        val_features = []
    cold_texts = [cold_item_text[item] for item in cold_item_ids]
    cold_bucket_ids = [text_bucket_lookup.get(item, "unknown") for item in cold_item_ids]
    cold_features = featurizer.transform(
        cold_texts,
        bucket_ids=cold_bucket_ids,
        log_prefix="[text][cold]",
    )
    _log_tokenizer_bucket_stats(featurizer, "cold_tokenized")
    print(f"Warm text features shape: ({len(warm_features)}, {len(warm_features[0]) if warm_features else 0})")
    if val_features:
        print(f"Val text features shape: ({len(val_features)}, {len(val_features[0])})")
    print(f"Cold text features shape: ({len(cold_features)}, {len(cold_features[0]) if cold_features else 0})")

    out_path = Path(out_dir)
    data_io.save_json(warm_item_ids, out_path / "warm_item_ids.json")
    if val_item_ids:
        data_io.save_json(val_item_ids, out_path / "val_item_ids.json")
    data_io.save_json(cold_item_ids, out_path / "cold_item_ids.json")
    data_io.save_matrix(warm_features, out_path / "warm_item_text_features.json")
    if val_features:
        data_io.save_matrix(val_features, out_path / "val_item_text_features.json")
    data_io.save_matrix(cold_features, out_path / "cold_item_text_features.json")
    encoder_state = {
        "encoder_type": encoder_type,
        "state": featurizer.save_state(),
    }
    data_io.save_json(encoder_state, out_path / "text_encoder_state.json")
    data_io.save_json(encoder_state, out_path / "tfidf_state.json")


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
    k_eval: int | Sequence[int] = 10,
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
    micm_cfg: dict | None = None,
    cmcl_cfg: dict | None = None,
    mf_cfg: dict | None = None,
    backend: str = "numpy",
    prefer_gpu: bool = True,
) -> Dict[str, Dict[str, object]]:
    data_path = Path(data_dir)
    warm_rows = data_io.load_interactions(data_path / "warm_interactions.csv")
    cold_rows = data_io.load_interactions(data_path / "cold_interactions.csv")
    warm_item_ids = data_io.load_json(data_path / "warm_item_ids.json")
    cold_item_ids = data_io.load_json(data_path / "cold_item_ids.json")
    warm_features = data_io.load_matrix(data_path / "warm_item_text_features.json")
    cold_features = data_io.load_matrix(data_path / "cold_item_text_features.json")
    warm_item_text = _unique_item_text(warm_rows)
    cold_item_text = _unique_item_text(cold_rows)
    text_len_lookup: Dict[str, int] = {}
    for source in (warm_rows, cold_rows):
        for row in source:
            text_len = row.get("text_len")
            if text_len is None:
                continue
            item_id = row["item_id"]
            if item_id not in text_len_lookup:
                text_len_lookup[item_id] = int(text_len)
    text_thresholds = _quantile_thresholds(list(text_len_lookup.values()), (0.33, 0.66))
    text_labels = ["short", "medium", "long"]
    all_item_ids = warm_item_ids + cold_item_ids
    text_bucket_lookup = {
        item_id: _assign_bucket(text_len_lookup.get(item_id, 0), text_thresholds, text_labels)
        for item_id in all_item_ids
    }
    _log_bucket_text_coverage("warm_text", warm_item_ids, warm_item_text, text_bucket_lookup)
    _log_bucket_text_coverage("cold_text", cold_item_ids, cold_item_text, text_bucket_lookup)

    if isinstance(k_eval, int):
        eval_ks = [k_eval]
    else:
        eval_ks = [int(k) for k in k_eval]
    if not eval_ks or any(k <= 0 for k in eval_ks):
        raise ValueError("k_eval must contain positive integer(s).")

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

    available_models = ("ctrlite", "a2f", "ctpf", "cdl", "hft", "micm", "cmcl")
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

    results: Dict[str, Dict[str, object]] = {}
    ctrlite_cache: Dict[str, object] = {}
    infer_batch_size = max(1, int(mf_cfg.get("infer_batch_size", 4096)))
    score_batch_size = max(1, int(mf_cfg.get("score_batch_size", 4096)))

    def _score(users: list[list[float]], items: list[list[float]]) -> list[list[float]]:
        if use_torch:
            return torch_backend.score_users_on_cold(
                users, items, batch_size=score_batch_size, prefer_gpu=prefer_gpu
            )
        return ctr_lite.score_users_on_cold(users, items)

    user_history_counts: Dict[str, int] = {}
    for row in warm_rows:
        user_history_counts[row["user_id"]] = user_history_counts.get(row["user_id"], 0) + 1
    item_popularity_counts: Dict[str, int] = {}
    for row in warm_rows:
        item_popularity_counts[row["item_id"]] = item_popularity_counts.get(row["item_id"], 0) + 1
    for row in cold_rows:
        item_popularity_counts[row["item_id"]] = item_popularity_counts.get(row["item_id"], 0) + 1
    item_text_len_map: Dict[str, int] = {
        item_id: text_len_lookup.get(item_id, 0) for item_id in cold_item_ids if item_id in text_len_lookup
    }

    def _collect_metrics(score_matrix: list[list[float]]) -> Dict[str, object]:
        metrics: Dict[str, object] = {}
        aggregated_buckets: Dict[str, Dict[str, Dict[str, float]]] = {}
        evaluated = 0
        for k_value in eval_ks:
            per_k = hit_ndcg_at_k(
                score_matrix,
                user_to_idx,
                cold_item_ids,
                cold_rows,
                k=k_value,
                user_history=user_history_counts,
                item_popularity=item_popularity_counts,
                item_text_len=item_text_len_map,
            )
            metrics[f"hit@{k_value}"] = per_k.get(f"hit@{k_value}", 0.0)
            hit_ci = per_k.get(f"hit@{k_value}_ci")
            if hit_ci is not None:
                metrics[f"hit@{k_value}_ci"] = hit_ci
            metrics[f"ndcg@{k_value}"] = per_k.get(f"ndcg@{k_value}", 0.0)
            ndcg_ci = per_k.get(f"ndcg@{k_value}_ci")
            if ndcg_ci is not None:
                metrics[f"ndcg@{k_value}_ci"] = ndcg_ci
            evaluated = per_k.get("evaluated_users", evaluated or 0)
            bucket_info = per_k.get("buckets") or {}
            for bucket_name, labels in bucket_info.items():
                bucket_entry = aggregated_buckets.setdefault(bucket_name, {})
                for label, stats in labels.items():
                    label_entry = bucket_entry.setdefault(label, {})
                    for key, value in stats.items():
                        label_entry[key] = value
        metrics["evaluated_users"] = evaluated
        if aggregated_buckets:
            metrics["buckets"] = aggregated_buckets
        return metrics

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
            metrics = _collect_metrics(scores)
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
            metrics = _collect_metrics(scores)
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
            metrics = _collect_metrics(scores)
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
            metrics = _collect_metrics(scores)
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
            metrics = _collect_metrics(scores)
            results["hft"] = metrics
        elif name == "micm":
            if not use_torch:
                print("micm requires backend='torch'; skipping.")
                continue
            cfg = micm_cfg or {}
            pos_cfg = dict(cfg.get("positives", {}) or {})
            path_value = pos_cfg.get("path")
            if isinstance(path_value, str):
                path_value = path_value.strip() or None
            micm_config = torch_backend.MICMConfig(
                lr=float(cfg.get("lr", 1e-3)),
                reg=float(cfg.get("reg", 1e-4)),
                iters=_coerce_int(cfg.get("iters"), 200),
                batch_size=max(1, _coerce_int(cfg.get("batch_size"), 1024)),
                temperature=float(cfg.get("temperature", 0.07)),
                symmetric=_coerce_bool(cfg.get("symmetric"), True),
                loss_type=str(cfg.get("loss_type", "info_nce")),
                num_workers=max(0, _coerce_int(cfg.get("num_workers"), 0)),
                positives=torch_backend.MICMPositivesConfig(
                    path=path_value,
                    k=max(1, _coerce_int(pos_cfg.get("k"), 5)),
                    min_cos=_coerce_optional_float(pos_cfg.get("min_cos")),
                    include_self=_coerce_bool(pos_cfg.get("include_self"), True),
                    whiten_cf=_coerce_bool(pos_cfg.get("whiten_cf"), False),
                ),
            )
            micm_model = torch_backend.train_micm(
                warm_item_ids,
                warm_features,
                V_ordered,
                micm_config,
                prefer_gpu=prefer_gpu,
            )
            V_cold = torch_backend.infer_micm(
                micm_model,
                cold_features,
                prefer_gpu=prefer_gpu,
                batch_size=infer_batch_size,
            )
            scores = _score(U, V_cold)
            metrics = _collect_metrics(scores)
            results["micm"] = metrics
        elif name == "cmcl":
            if not use_torch:
                print("cmcl requires backend='torch'; skipping.")
                continue
            cfg = cmcl_cfg or {}
            exposure_cfg = dict(cfg.get("exposure", {}) or {})
            exposure_path_value = exposure_cfg.get("path")
            exposure_path = (
                Path(exposure_path_value)
                if exposure_path_value
                else models_dir / "exposure.ckpt"
            )
            max_pairs_value = exposure_cfg.get("max_training_samples")
            exposure_config = ExposureConfig(
                negatives_per_positive=max(1, _coerce_int(exposure_cfg.get("negatives_per_positive"), 5)),
                hidden_dim=max(1, _coerce_int(exposure_cfg.get("hidden_dim"), 64)),
                batch_size=max(32, _coerce_int(exposure_cfg.get("batch_size"), 4096)),
                epochs=max(1, _coerce_int(exposure_cfg.get("epochs"), 5)),
                lr=float(exposure_cfg.get("lr", 1e-3)),
                pi_min=float(exposure_cfg.get("pi_min", 0.01)),
                seed=_coerce_int(exposure_cfg.get("seed"), 13),
                max_training_samples=(
                    None
                    if max_pairs_value in {None, "", "none"}
                    else max(1, _coerce_int(max_pairs_value, 1_000_000))
                ),
                prefer_gpu=prefer_gpu,
            )
            if not exposure_path.exists():
                print(f"[cmcl] exposure checkpoint missing at {exposure_path}; training estimator.")
                train_exposure_model(warm_rows, exposure_path, exposure_config)
            exposure_state = load_exposure_checkpoint(exposure_path)
            pi_lookup = exposure_state.get("pi_lookup", {})
            hard_cfg = dict(cfg.get("hard_negatives", {}) or {})
            cmcl_config = torch_backend.CMCLConfig(
                lr=float(cfg.get("lr", 5e-4)),
                reg=float(cfg.get("reg", 1e-4)),
                iters=max(1, _coerce_int(cfg.get("iters"), 60)),
                batch_size=max(1, _coerce_int(cfg.get("batch_size"), 128)),
                temperature=float(cfg.get("temperature", 0.05)),
                self_normalize=_coerce_bool(cfg.get("self_normalize"), True),
                topk_focal_k=max(0, _coerce_int(cfg.get("topk_focal_k"), 0)),
                topk_focal_gamma=float(cfg.get("topk_focal_gamma", 1.0)),
                max_positives=max(0, _coerce_int(cfg.get("max_positives"), 0)),
                pi_floor=float(cfg.get("pi_floor", exposure_config.pi_min)),
                max_weight=float(cfg.get("max_weight", 50.0)),
                encoder_lr=_coerce_optional_float(cfg.get("encoder_lr")),
                head_lr=_coerce_optional_float(cfg.get("head_lr")),
                weight_decay=_coerce_optional_float(cfg.get("weight_decay")),
                proj_hidden_dim=max(32, _coerce_int(cfg.get("proj_hidden_dim"), 512)),
                dropout=float(cfg.get("dropout", 0.1)),
                grad_alert_patience=max(1, _coerce_int(cfg.get("grad_alert_patience"), 5)),
                log_interval=max(0, _coerce_int(cfg.get("log_interval"), 10)),
                text_log_batches=max(0, _coerce_int(cfg.get("text_log_batches"), 5)),
                sampled_negatives=max(0, _coerce_int(cfg.get("sampled_negatives"), 256)),
                neg_sampler_seed=_coerce_int(cfg.get("neg_sampler_seed"), seed),
                margin_m=float(cfg.get("margin_m", 0.1)),
                margin_alpha=float(cfg.get("margin_alpha", 0.05)),
                hard_negatives=torch_backend.CMCLHardNegativesConfig(
                    k=max(0, _coerce_int(hard_cfg.get("k"), 0)),
                    min_sim=float(hard_cfg.get("min_sim", 0.5)),
                ),
            )
            item_exposure = torch_backend.aggregate_item_exposure(pi_lookup, cmcl_config.pi_floor)
            cmcl_model = torch_backend.train_cmcl(
                warm_rows,
                warm_item_ids,
                warm_features,
                U,
                user_to_idx,
                pi_lookup,
                cmcl_config,
                text_len_lookup,
                text_bucket_lookup,
                item_exposure,
                prefer_gpu=prefer_gpu,
            )
            V_cold = torch_backend.infer_cmcl(
                cmcl_model,
                cold_features,
                prefer_gpu=prefer_gpu,
                batch_size=infer_batch_size,
            )
            adapted_users = torch_backend.adapt_users_with_cmcl(
                cmcl_model,
                U,
                prefer_gpu=prefer_gpu,
                batch_size=score_batch_size,
            )
            scores = _score(adapted_users, V_cold)
            pop_bias_cfg = cfg.get("pop_bias", {})
            if pop_bias_cfg.get("enable"):
                head_frac = float(pop_bias_cfg.get("head_frac", 0.2))
                mid_frac = float(pop_bias_cfg.get("mid_frac", 0.3))
                beta_tail = float(pop_bias_cfg.get("beta_tail", 0.05))
                gamma_pop = float(pop_bias_cfg.get("gamma_pop", -0.02))
                bias_vector = _build_popularity_bias(
                    cold_item_ids,
                    item_popularity_counts,
                    head_frac,
                    mid_frac,
                    beta_tail,
                    gamma_pop,
                )
                if bias_vector:
                    for row in scores:
                        for idx, bias in enumerate(bias_vector):
                            row[idx] += bias
            metrics = _collect_metrics(scores)
            results["cmcl"] = metrics

    baseline = results.get("micm")
    if baseline:
        baseline_buckets = baseline.get("buckets") if isinstance(baseline, dict) else None
        for model_name, metrics in results.items():
            if model_name == "micm":
                continue
            deltas: Dict[str, object] = {}
            for key, value in metrics.items():
                base_value = baseline.get(key) if isinstance(baseline, dict) else None
                if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
                    deltas[key] = value - base_value
            model_buckets = metrics.get("buckets") if isinstance(metrics, dict) else None
            if baseline_buckets and isinstance(model_buckets, dict):
                bucket_delta: Dict[str, Dict[str, Dict[str, float]]] = {}
                for bucket_name, labels in model_buckets.items():
                    base_labels = baseline_buckets.get(bucket_name)
                    if not isinstance(base_labels, dict):
                        continue
                    label_delta: Dict[str, Dict[str, float]] = {}
                    for label, stats in labels.items():
                        base_stats = base_labels.get(label)
                        if not isinstance(base_stats, dict):
                            continue
                        stat_delta: Dict[str, float] = {}
                        for stat_key, stat_value in stats.items():
                            base_stat_value = base_stats.get(stat_key)
                            if isinstance(stat_value, (int, float)) and isinstance(base_stat_value, (int, float)):
                                stat_delta[stat_key] = stat_value - base_stat_value
                        if stat_delta:
                            label_delta[label] = stat_delta
                    if label_delta:
                        bucket_delta[bucket_name] = label_delta
                if bucket_delta:
                    deltas["buckets"] = bucket_delta
            if deltas:
                metrics["delta_vs_micm"] = deltas

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
        adaptive_metrics = _collect_metrics(scores_adapt)
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
