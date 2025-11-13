"""Pseudo-exposure estimation for counterfactual training."""
from __future__ import annotations

import bisect
import gc
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn


def _device(prefer_gpu: bool) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class ExposureConfig:
    negatives_per_positive: int = 5
    hidden_dim: int = 64
    batch_size: int = 4096
    epochs: int = 5
    lr: float = 1e-3
    pi_min: float = 0.01
    seed: int = 13
    max_training_samples: int | None = None
    prefer_gpu: bool = True


MAX_EXPOSURE_TRAIN_EXAMPLES = 100_000


def _cap_training_examples(
    features: List[List[float]],
    labels: List[float],
    seed: int,
) -> tuple[List[List[float]], List[float]]:
    if len(features) <= MAX_EXPOSURE_TRAIN_EXAMPLES:
        return features, labels
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(features)), MAX_EXPOSURE_TRAIN_EXAMPLES))
    capped_features = [features[idx] for idx in indices]
    capped_labels = [labels[idx] for idx in indices]
    print(
        f"[Exposure] capped training pairs to {len(capped_features)} "
        f"(max={MAX_EXPOSURE_TRAIN_EXAMPLES})."
    )
    return capped_features, capped_labels


class _ExposureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Sequential(nn.Linear(input_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def _collect_item_metadata(rows: Sequence[dict]) -> Tuple[Dict[str, dict], List[str], int]:
    meta: Dict[str, dict] = {}
    genre_tokens: set[str] = set()
    max_text_len = 0
    for row in rows:
        item = row["item_id"]
        if item in meta:
            continue
        genres = [token for token in str(row.get("item_genres") or "").split() if token]
        genre_tokens.update(genres)
        text_len = int(row.get("text_len") or 0)
        max_text_len = max(max_text_len, text_len)
        release_ts = row.get("release_ts")
        release_ts = int(release_ts) if release_ts not in {None, ""} else 0
        meta[item] = {
            "release_ts": max(release_ts, 0),
            "text_len": text_len,
            "genres": genres,
        }
    return meta, sorted(genre_tokens), max(max_text_len, 1)


def _popularity_quantiles(rows: Sequence[dict]) -> Dict[str, float]:
    counts = Counter(row["item_id"] for row in rows)
    if not counts:
        return {}
    ordered = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    denom = len(ordered) - 1
    return {item: idx / denom for idx, (item, _) in enumerate(ordered)}


class _ExposureBuilder:
    def __init__(self, warm_rows: Sequence[dict], config: ExposureConfig) -> None:
        self.config = config
        self.warm_rows = sorted(
            warm_rows,
            key=lambda row: (int(row.get("timestamp") or 0), row["user_id"], row["item_id"]),
        )
        self.item_meta, self.genre_vocab, self.max_text_len = _collect_item_metadata(warm_rows)
        self.genre_index = {genre: idx for idx, genre in enumerate(self.genre_vocab)}
        self.item_popularity = _popularity_quantiles(warm_rows)
        self.user_totals = Counter(row["user_id"] for row in warm_rows)
        self.release_pairs = sorted(
            (self.item_meta[item]["release_ts"], item) for item in self.item_meta
        )
        self.release_times = [pair[0] for pair in self.release_pairs]
        self.release_items = [pair[1] for pair in self.release_pairs]
        self.feature_dim = 3 + len(self.genre_vocab) + 1 + len(self.genre_vocab)
        self._positives = 0
        self._negatives = 0

    def _item_features(self, item_id: str, timestamp: int) -> List[float]:
        meta = self.item_meta.get(item_id)
        if meta is None:
            return [0.0] * (3 + len(self.genre_vocab))
        popularity = self.item_popularity.get(item_id, 0.0)
        recency_seconds = max(timestamp - meta["release_ts"], 0)
        recency_days = recency_seconds / 86400.0
        recency = math.log1p(max(recency_days, 0.0))
        text_norm = min(meta["text_len"] / self.max_text_len, 1.0)
        genre_vec = [1.0 if genre in meta["genres"] else 0.0 for genre in self.genre_vocab]
        return [popularity, recency, text_norm] + genre_vec

    def _user_features(self, user_id: str, history_count: int, genres: Counter) -> List[float]:
        total_interactions = max(self.user_totals.get(user_id, 1), 1)
        activity_quantile = history_count / total_interactions
        total_genre = sum(genres.values())
        affinity = [
            (genres.get(genre, 0) / total_genre) if total_genre > 0 else 0.0
            for genre in self.genre_vocab
        ]
        return [activity_quantile] + affinity

    def _concat(self, user_vec: List[float], item_vec: List[float]) -> List[float]:
        return item_vec + user_vec

    def _sample_negatives(
        self,
        timestamp: int,
        consumed: set[str],
        rng: random.Random,
    ) -> List[str]:
        cutoff = bisect.bisect_right(self.release_times, timestamp)
        if cutoff == 0:
            return []
        sampled: List[str] = []
        candidates = self.release_items[:cutoff]
        max_trials = max(50, 10 * self.config.negatives_per_positive)
        trials = 0
        while len(sampled) < self.config.negatives_per_positive and trials < max_trials:
            candidate = candidates[rng.randrange(cutoff)]
            trials += 1
            if candidate in consumed or candidate in sampled:
                continue
            sampled.append(candidate)
        return sampled

    def build_training_pairs(self) -> Tuple[List[List[float]], List[float]]:
        rng = random.Random(self.config.seed)
        user_history: Dict[str, int] = defaultdict(int)
        user_genres: Dict[str, Counter] = defaultdict(Counter)
        user_consumed: Dict[str, set[str]] = defaultdict(set)
        features: List[List[float]] = []
        labels: List[float] = []

        for row in self.warm_rows:
            user = row["user_id"]
            item = row["item_id"]
            timestamp = int(row.get("timestamp") or 0)
            user_vec = self._user_features(user, user_history[user], user_genres[user])
            item_vec = self._item_features(item, timestamp)
            features.append(self._concat(user_vec, item_vec))
            labels.append(1.0)
            self._positives += 1

            neg_items = self._sample_negatives(timestamp, user_consumed[user], rng)
            for neg in neg_items:
                if self.config.max_training_samples is not None and len(labels) >= self.config.max_training_samples:
                    break
                neg_vec = self._item_features(neg, timestamp)
                features.append(self._concat(user_vec, neg_vec))
                labels.append(0.0)
                self._negatives += 1

            user_history[user] += 1
            for genre in self.item_meta.get(item, {}).get("genres", []):
                user_genres[user][genre] += 1
            user_consumed[user].add(item)

        print(
            f"Exposure pairs: {self._positives} positives, "
            f"{self._negatives} negatives (k={self.config.negatives_per_positive})."
        )
        return features, labels

    def compute_pi_lookup(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        model.eval()
        user_history: Dict[str, int] = defaultdict(int)
        user_genres: Dict[str, Counter] = defaultdict(Counter)
        pi_lookup: Dict[str, float] = {}
        batch_vectors: List[List[float]] = []
        batch_keys: List[Tuple[str, str]] = []
        batch_size = 8192

        def _flush() -> None:
            if not batch_vectors:
                return
            with torch.no_grad():
                inputs = torch.tensor(batch_vectors, dtype=torch.float32, device=device)
                preds = model(inputs).clamp(self.config.pi_min, 1.0 - self.config.pi_min)
                for key, value in zip(batch_keys, preds.squeeze(1).tolist()):
                    pi_lookup[f"{key[0]}::{key[1]}"] = float(value)
            batch_vectors.clear()
            batch_keys.clear()

        for row in self.warm_rows:
            user = row["user_id"]
            item = row["item_id"]
            timestamp = int(row.get("timestamp") or 0)
            user_vec = self._user_features(user, user_history[user], user_genres[user])
            item_vec = self._item_features(item, timestamp)
            batch_vectors.append(self._concat(user_vec, item_vec))
            batch_keys.append((user, item))
            if len(batch_vectors) >= batch_size:
                _flush()
            user_history[user] += 1
            for genre in self.item_meta.get(item, {}).get("genres", []):
                user_genres[user][genre] += 1
        _flush()
        return pi_lookup


def train_exposure_model(
    warm_rows: Sequence[dict],
    out_path: str | Path,
    config: ExposureConfig,
) -> Dict[str, float]:
    builder = _ExposureBuilder(warm_rows, config)
    features, labels = builder.build_training_pairs()
    features, labels = _cap_training_examples(features, labels, config.seed)
    if not features:
        raise RuntimeError("No exposure pairs could be generated.")

    device = _device(config.prefer_gpu)
    inputs = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    del features, labels
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    requested_batch = max(32, config.batch_size)
    effective_batch = min(256, requested_batch)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        persistent_workers=False,
        pin_memory=False,
    )

    model = _ExposureMLP(builder.feature_dim, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCELoss()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        avg_loss = epoch_loss / max(1, len(loader))
        print(f"[Exposure] epoch {epoch + 1}/{config.epochs} - loss={avg_loss:.4f}")

    pi_lookup = builder.compute_pi_lookup(model, device)
    checkpoint = {
        "config": asdict(config),
        "genre_vocab": builder.genre_vocab,
        "item_popularity": builder.item_popularity,
        "max_text_len": builder.max_text_len,
        "feature_dim": builder.feature_dim,
        "positives": builder._positives,
        "negatives": builder._negatives,
        "pi_lookup": pi_lookup,
        "model_state": model.cpu().state_dict(),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    print(f"Saved exposure checkpoint with {len(pi_lookup)} entries to {out_path}.")
    del dataset, loader, inputs, targets, model, optimizer, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pi_lookup


def load_exposure_checkpoint(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu")


__all__ = [
    "ExposureConfig",
    "train_exposure_model",
    "load_exposure_checkpoint",
]
