"""Evaluation helpers for cold-start ranking."""
from __future__ import annotations

import math
import random
from statistics import mean
from typing import Dict, Sequence, Tuple, List


def group_positives_by_user(interactions: Sequence[dict]) -> Dict[str, set[str]]:
    grouped: Dict[str, set[str]] = {}
    for row in interactions:
        grouped.setdefault(row["user_id"], set()).add(row["item_id"])
    return grouped


def _bootstrap_ci(values: Sequence[float], samples: int = 200, confidence: float = 0.95) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(0)
    n = len(values)
    estimates: List[float] = []
    for _ in range(samples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(resample) / n)
    estimates.sort()
    lower_idx = max(0, int((1 - confidence) / 2 * len(estimates)))
    upper_idx = min(len(estimates) - 1, int((confidence + (1 - confidence) / 2) * len(estimates)) - 1)
    return estimates[lower_idx], estimates[upper_idx]


def _quantile_thresholds(values: Sequence[int], quantiles: Sequence[float]) -> List[float]:
    if not values:
        return []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    thresholds: List[float] = []
    for q in quantiles:
        if n == 1:
            thresholds.append(sorted_vals[0])
            continue
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        thresholds.append(sorted_vals[idx])
    return thresholds


def _assign_bucket(value: float, thresholds: Sequence[float], labels: Sequence[str]) -> str:
    if not thresholds or not labels:
        return labels[0] if labels else "all"
    for threshold, label in zip(thresholds, labels):
        if value <= threshold:
            return label
    return labels[-1]


def hit_ndcg_at_k(
    scores: list[list[float]],
    user_to_idx: Dict[str, int],
    cold_item_ids: Sequence[str],
    cold_interactions: Sequence[dict],
    k: int = 10,
    user_history: Dict[str, int] | None = None,
    item_popularity: Dict[str, int] | None = None,
    item_text_len: Dict[str, int] | None = None,
    bootstrap_samples: int = 200,
) -> Dict[str, object]:
    positives = group_positives_by_user(cold_interactions)
    if not positives:
        return {
            f"hit@{k}": 0.0,
            f"hit@{k}_ci": (0.0, 0.0),
            f"ndcg@{k}": 0.0,
            f"ndcg@{k}_ci": (0.0, 0.0),
            "evaluated_users": 0,
        }

    user_history = user_history or {}
    item_popularity = item_popularity or {}
    item_text_len = item_text_len or {}

    user_hits: List[float] = []
    user_ndcgs: List[float] = []

    bucket_accumulators: Dict[str, Dict[str, Dict[str, float]]] = {}
    bucket_counts: Dict[str, Dict[str, int]] = {}

    item_text_thresholds = _quantile_thresholds(list(item_text_len.values()), (0.33, 0.66))
    item_pop_thresholds = _quantile_thresholds(list(item_popularity.values()), (0.33, 0.66))
    user_hist_thresholds = _quantile_thresholds(
        [user_history.get(user, 0) for user in positives.keys()],
        (0.33, 0.66),
    )

    text_labels = ["short", "medium", "long"]
    pop_labels = ["cold", "mid", "hot"]
    history_labels = ["sparse", "moderate", "heavy"]

    item_text_bucket = {
        item_id: _assign_bucket(item_text_len.get(item_id, 0), item_text_thresholds, text_labels)
        for item_id in cold_item_ids
    }
    item_pop_bucket = {
        item_id: _assign_bucket(item_popularity.get(item_id, 0), item_pop_thresholds, pop_labels)
        for item_id in cold_item_ids
    }
    user_history_bucket = {
        user_id: _assign_bucket(user_history.get(user_id, 0), user_hist_thresholds, history_labels)
        for user_id in positives.keys()
    }

    for user_id, pos_items in positives.items():
        if user_id not in user_to_idx:
            continue
        if not pos_items:
            continue
        user_idx = user_to_idx[user_id]
        user_scores = scores[user_idx]
        order = sorted(range(len(user_scores)), key=lambda idx: user_scores[idx], reverse=True)[:k]
        ranked_items = [cold_item_ids[i] for i in order]
        hit = 1.0 if any(item in pos_items for item in ranked_items) else 0.0
        dcg = 0.0
        for rank, item in enumerate(ranked_items, start=1):
            if item in pos_items:
                dcg += 1.0 / math.log2(rank + 1)
        ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, min(len(pos_items), k) + 1))
        ndcg = dcg / ideal if ideal > 0 else 0.0
        user_hits.append(hit)
        user_ndcgs.append(ndcg)

        # Item bucket metrics
        for bucket_name, (bucket_map, labels) in {
            "item_text_len": (item_text_bucket, text_labels),
            "item_popularity": (item_pop_bucket, pop_labels),
        }.items():
            per_bucket = bucket_accumulators.setdefault(bucket_name, {})
            per_bucket_counts = bucket_counts.setdefault(bucket_name, {})
            grouped: Dict[str, set[str]] = {}
            for item in pos_items:
                label = bucket_map.get(item)
                if label:
                    grouped.setdefault(label, set()).add(item)
            for label, items_in_bucket in grouped.items():
                bucket_hit = 1.0 if any(item in items_in_bucket for item in ranked_items) else 0.0
                bucket_dcg = 0.0
                for rank, item in enumerate(ranked_items, start=1):
                    if item in items_in_bucket:
                        bucket_dcg += 1.0 / math.log2(rank + 1)
                ideal_bucket = sum(
                    1.0 / math.log2(idx + 1) for idx in range(1, min(len(items_in_bucket), k) + 1)
                )
                bucket_ndcg = bucket_dcg / ideal_bucket if ideal_bucket > 0 else 0.0
                stats = per_bucket.setdefault(label, {"hit": 0.0, "ndcg": 0.0})
                stats["hit"] += bucket_hit
                stats["ndcg"] += bucket_ndcg
                per_bucket_counts[label] = per_bucket_counts.get(label, 0) + 1

        # User history buckets
        history_label = user_history_bucket.get(user_id)
        if history_label:
            per_bucket = bucket_accumulators.setdefault("user_history_len", {})
            counts = bucket_counts.setdefault("user_history_len", {})
            stats = per_bucket.setdefault(history_label, {"hit": 0.0, "ndcg": 0.0})
            stats["hit"] += hit
            stats["ndcg"] += ndcg
            counts[history_label] = counts.get(history_label, 0) + 1

    mean_hit = mean(user_hits) if user_hits else 0.0
    mean_ndcg = mean(user_ndcgs) if user_ndcgs else 0.0
    hit_ci = _bootstrap_ci(user_hits, samples=bootstrap_samples)
    ndcg_ci = _bootstrap_ci(user_ndcgs, samples=bootstrap_samples)

    bucket_outputs: Dict[str, Dict[str, Dict[str, float]]] = {}
    for bucket_name, metrics in bucket_accumulators.items():
        counts = bucket_counts.get(bucket_name, {})
        label_outputs: Dict[str, Dict[str, float]] = {}
        for label, stats in metrics.items():
            count = counts.get(label, 0)
            if count <= 0:
                continue
            label_outputs[label] = {
                f"hit@{k}": stats["hit"] / count,
                f"ndcg@{k}": stats["ndcg"] / count,
            }
        if label_outputs:
            bucket_outputs[bucket_name] = label_outputs

    result: Dict[str, object] = {
        f"hit@{k}": mean_hit,
        f"hit@{k}_ci": hit_ci,
        f"ndcg@{k}": mean_ndcg,
        f"ndcg@{k}_ci": ndcg_ci,
        "evaluated_users": len(user_hits),
    }
    if bucket_outputs:
        result["buckets"] = bucket_outputs
    return result
