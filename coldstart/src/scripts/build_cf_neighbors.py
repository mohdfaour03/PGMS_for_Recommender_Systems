"""Build CF-based positive sets for MICM training."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def _load_warm_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of warm item ids in {path}")
    return [str(item) for item in data]


def _load_item_factors(path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append([float(part) for part in stripped.split(",")])
    if not rows:
        raise ValueError(f"No item factors found in {path}")
    return np.asarray(rows, dtype=np.float32)


def _zca_whiten(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    u, s, _ = np.linalg.svd(cov, full_matrices=False)
    transform = u @ np.diag(1.0 / np.sqrt(s + eps)) @ u.T
    whitened = centered @ transform
    return whitened.astype(np.float32)


def _chunked_cosine_topk(
    vectors: torch.Tensor,
    k: int,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = vectors.size(0)
    device = vectors.device
    indices = torch.empty((n, k), dtype=torch.long, device=device)
    scores = torch.empty((n, k), dtype=torch.float32, device=device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = vectors[start:end]
        sim = chunk @ vectors.T
        row_indices = torch.arange(start, end, device=device)
        sim[torch.arange(end - start), row_indices] = -math.inf
        top_scores, top_idx = torch.topk(sim, k=k, dim=1)
        indices[start:end] = top_idx
        scores[start:end] = top_scores
    return indices, scores


def _build_neighbor_map(
    warm_ids: Sequence[str],
    factors: np.ndarray,
    k: int,
    min_cos: float | None,
    include_self: bool,
    whiten: bool,
    chunk_size: int,
    seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[List[str], List[float]]]]:
    if whiten:
        factors = _zca_whiten(factors)
    norms = np.linalg.norm(factors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = factors / norms
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors = torch.tensor(normalized, dtype=torch.float32, device=torch_device)
    vectors = torch.nn.functional.normalize(vectors, dim=1)
    top_idx, top_scores = _chunked_cosine_topk(vectors, k, chunk_size=chunk_size)
    neighbor_map: Dict[str, List[str]] = {}
    debug_payload: Dict[str, Tuple[List[str], List[float]]] = {}
    min_cos_threshold = -1.1 if min_cos is None else float(min_cos)
    rng = random.Random(seed)
    debug_candidates = set(rng.sample(range(len(warm_ids)), min(5, len(warm_ids))))
    for row_idx, item_id in enumerate(warm_ids):
        indices = top_idx[row_idx].tolist()
        scores = top_scores[row_idx].tolist()
        filtered: List[str] = []
        filtered_scores: List[float] = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(warm_ids):
                continue
            if score < min_cos_threshold:
                continue
            candidate_id = warm_ids[idx]
            if candidate_id == item_id:
                continue
            filtered.append(candidate_id)
            filtered_scores.append(score)
            if len(filtered) >= k:
                break
        if include_self:
            filtered = [item_id] + filtered
            filtered_scores = [1.0] + filtered_scores
        elif not filtered:
            filtered = [item_id]
            filtered_scores = [1.0]
        neighbor_map[item_id] = filtered[: k + (1 if include_self else 0)]
        if row_idx in debug_candidates:
            debug_payload[item_id] = (filtered[: k + 1], filtered_scores[: k + 1])
    return neighbor_map, debug_payload


def _save_json(data: Dict[str, Iterable[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CF nearest neighbors for MICM positives.")
    parser.add_argument("--item-factors", type=Path, required=True, help="Path to V_warm.json factors.")
    parser.add_argument("--warm-ids", type=Path, required=True, help="Path to warm_item_ids.json.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/positives/cf_knn_top5.json"),
        help="Destination JSON for positive neighbors.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors to keep (excludes self).")
    parser.add_argument(
        "--min-cos",
        type=float,
        default=0.6,
        help="Minimum cosine similarity to keep. Set negative to disable.",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include the anchor itself as a positive.",
    )
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="Apply ZCA whitening before normalisation.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Chunk size for batched similarity computation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when selecting debug examples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warm_ids = _load_warm_ids(args.warm_ids)
    factors = _load_item_factors(args.item_factors)
    if len(warm_ids) != factors.shape[0]:
        raise ValueError(
            f"Warm id count ({len(warm_ids)}) does not match factor rows ({factors.shape[0]})."
        )
    min_cos = None if args.min_cos is None or args.min_cos < 0 else args.min_cos
    neighbor_map, debug_payload = _build_neighbor_map(
        warm_ids,
        factors,
        k=max(1, args.k),
        min_cos=min_cos,
        include_self=args.include_self,
        whiten=args.whiten,
        chunk_size=max(256, args.chunk_size),
        seed=args.seed,
    )
    coverage = sum(bool(neighbors) for neighbors in neighbor_map.values()) / max(len(neighbor_map), 1)
    if coverage < 0.95:
        raise RuntimeError(
            f"Neighbor coverage below 95%: {coverage * 100:.2f}%."
        )
    _save_json(neighbor_map, args.output)
    print(f"Saved neighbor map with {len(neighbor_map)} items to {args.output}")
    for anchor, (pos_ids, scores) in debug_payload.items():
        formatted = ", ".join(f"{pid}:{score:.3f}" for pid, score in zip(pos_ids, scores))
        print(f"[debug] {anchor} -> {formatted}")


if __name__ == "__main__":
    main()
