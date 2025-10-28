"""Evaluation helpers for cold-start ranking."""
from __future__ import annotations

import math
from statistics import mean
from typing import Dict, Sequence


def group_positives_by_user(interactions: Sequence[dict]) -> Dict[str, set[str]]:
    grouped: Dict[str, set[str]] = {}
    for row in interactions:
        grouped.setdefault(row["user_id"], set()).add(row["item_id"])
    return grouped


def hit_ndcg_at_k(
    scores: list[list[float]],
    user_to_idx: Dict[str, int],
    cold_item_ids: Sequence[str],
    cold_interactions: Sequence[dict],
    k: int = 10,
) -> Dict[str, float]:
    positives = group_positives_by_user(cold_interactions)
    if not positives:
        return {f"hit@{k}": 0.0, f"ndcg@{k}": 0.0, "evaluated_users": 0}

    hits: list[float] = []
    ndcgs: list[float] = []
    for user_id, pos_items in positives.items():
        if user_id not in user_to_idx:
            continue
        user_idx = user_to_idx[user_id]
        user_scores = scores[user_idx]
        order = sorted(
            range(len(user_scores)), key=lambda idx: user_scores[idx], reverse=True
        )[:k]
        ranked_items = [cold_item_ids[i] for i in order]
        hits.append(1.0 if any(item in pos_items for item in ranked_items) else 0.0)

        dcg = 0.0
        for rank, item in enumerate(ranked_items, start=1):
            if item in pos_items:
                dcg += 1.0 / math.log2(rank + 1)
        ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, min(len(pos_items), k) + 1))
        ndcg = dcg / ideal if ideal > 0 else 0.0
        ndcgs.append(ndcg)

    return {
        f"hit@{k}": mean(hits) if hits else 0.0,
        f"ndcg@{k}": mean(ndcgs) if ndcgs else 0.0,
        "evaluated_users": len(hits),
    }
