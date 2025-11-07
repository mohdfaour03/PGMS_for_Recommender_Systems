"""PyTorch-backed training utilities with optional GPU acceleration."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


def _torch_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_index(values: Iterable[str]) -> Tuple[Dict[str, int], List[str]]:
    unique = sorted(set(values))
    mapping = {val: idx for idx, val in enumerate(unique)}
    return mapping, unique


class _InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, users: Sequence[int], items: Sequence[int], ratings: Sequence[float]) -> None:
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self) -> int:
        return self.users.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]


class _MFModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, factors: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, factors)
        self.item_emb = nn.Embedding(num_items, factors)
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_emb(users)
        item_vec = self.item_emb(items)
        return (user_vec * item_vec).sum(dim=1)


def train_mf(
    interactions: Sequence[dict],
    factors: int,
    reg: float,
    iters: int,
    lr: float,
    seed: int,
    batch_size: int = 4096,
    prefer_gpu: bool = True,
) -> Tuple[List[List[float]], List[List[float]], Dict[str, int], Dict[str, int]]:
    """Train matrix factorisation with PyTorch embeddings."""
    _set_seed(seed)
    device = _torch_device(prefer_gpu)

    user_to_idx, user_ids = _build_index(row["user_id"] for row in interactions)
    item_to_idx, item_ids = _build_index(row["item_id"] for row in interactions)

    users = [user_to_idx[row["user_id"]] for row in interactions]
    items = [item_to_idx[row["item_id"]] for row in interactions]
    ratings = [float(row["rating_or_y"]) for row in interactions]

    dataset = _InteractionDataset(users, items, ratings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = _MFModel(len(user_ids), len(item_ids), factors).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    reg_term = torch.tensor(reg, device=device, dtype=torch.float32)
    for _ in range(iters):
        for batch_users, batch_items, batch_ratings in loader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            batch_ratings = batch_ratings.to(device)

            optimizer.zero_grad()
            preds = model(batch_users, batch_items)
            loss = torch.nn.functional.mse_loss(preds, batch_ratings)
            loss += reg_term * (
                model.user_emb(batch_users).pow(2).mean() + model.item_emb(batch_items).pow(2).mean()
            )
            loss.backward()
            optimizer.step()

    U = model.user_emb.weight.detach().cpu().numpy().tolist()
    V = model.item_emb.weight.detach().cpu().numpy().tolist()
    return U, V, user_to_idx, item_to_idx


def _tensor_from(features: List[List[float]], device: torch.device) -> torch.Tensor:
    return torch.tensor(features, dtype=torch.float32, device=device)


def score_users_on_cold(
    user_factors: List[List[float]],
    cold_factors: List[List[float]],
    batch_size: int = 1024,
    prefer_gpu: bool = True,
) -> List[List[float]]:
    device = _torch_device(prefer_gpu)
    U = torch.tensor(user_factors, dtype=torch.float32, device=device)
    V = torch.tensor(cold_factors, dtype=torch.float32, device=device)
    scores: List[List[float]] = []
    with torch.no_grad():
        for start in range(0, U.shape[0], batch_size):
            block = U[start : start + batch_size]
            block_scores = block @ V.T
            scores.extend(block_scores.cpu().numpy().tolist())
    return scores


class CtrliteModel(nn.Module):
    def __init__(self, n_features: int, n_factors: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_factors, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class CtrliteTorchConfig:
    reg: float = 0.01
    lr: float = 0.1
    iters: int = 80
    batch_size: int = 1024


def train_ctrlite(
    warm_features: List[List[float]],
    warm_factors: List[List[float]],
    config: CtrliteTorchConfig,
    prefer_gpu: bool = True,
) -> CtrliteModel:
    device = _torch_device(prefer_gpu)
    model = CtrliteModel(len(warm_features[0]), len(warm_factors[0])).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.reg)

    X = _tensor_from(warm_features, device)
    Y = _tensor_from(warm_factors, device)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for _ in range(config.iters):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = torch.nn.functional.mse_loss(preds, batch_y)
            loss.backward()
            optimizer.step()
    return model.cpu()


def infer_ctrlite(
    model: CtrliteModel,
    cold_features: List[List[float]],
    prefer_gpu: bool = True,
    batch_size: int = 4096,
) -> List[List[float]]:
    device = _torch_device(prefer_gpu)
    model = model.to(device)
    model.eval()
    outputs: List[List[float]] = []
    with torch.no_grad():
        for start in range(0, len(cold_features), batch_size):
            batch = torch.tensor(
                cold_features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            preds = model(batch)
            outputs.extend(preds.cpu().numpy().tolist())
    model.cpu()
    return outputs


class A2FModel(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_factors: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_factors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class A2FTorchConfig:
    hidden_dim: int = 128
    lr: float = 0.01
    reg: float = 1e-4
    iters: int = 200
    batch_size: int = 512


def train_a2f(
    warm_features: List[List[float]],
    warm_factors: List[List[float]],
    config: A2FTorchConfig,
    prefer_gpu: bool = True,
) -> A2FModel:
    device = _torch_device(prefer_gpu)
    model = A2FModel(len(warm_features[0]), config.hidden_dim, len(warm_factors[0])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.reg)

    X = _tensor_from(warm_features, device)
    Y = _tensor_from(warm_factors, device)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for _ in range(config.iters):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = torch.nn.functional.mse_loss(preds, batch_y)
            loss.backward()
            optimizer.step()
    return model.cpu()


def infer_a2f(
    model: A2FModel,
    cold_features: List[List[float]],
    prefer_gpu: bool = True,
    batch_size: int = 4096,
) -> List[List[float]]:
    device = _torch_device(prefer_gpu)
    model = model.to(device)
    model.eval()
    outputs: List[List[float]] = []
    with torch.no_grad():
        for start in range(0, len(cold_features), batch_size):
            batch = torch.tensor(
                cold_features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            preds = model(batch)
            outputs.extend(preds.cpu().numpy().tolist())
    model.cpu()
    return outputs


class MICMModel(nn.Module):
    def __init__(self, n_features: int, n_factors: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, n_factors, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@dataclass
class MICMPositivesConfig:
    path: str | None = None
    k: int = 5
    min_cos: Optional[float] = 0.6
    include_self: bool = True
    whiten_cf: bool = False


@dataclass
class MICMConfig:
    lr: float = 1e-3
    reg: float = 1e-4
    iters: int = 200
    batch_size: int = 1024
    temperature: float = 0.07
    symmetric: bool = True
    loss_type: str = "info_nce"
    num_workers: int = 0
    positives: MICMPositivesConfig = field(default_factory=MICMPositivesConfig)


@dataclass
class CMCLHardNegativesConfig:
    k: int = 0
    min_sim: float = 0.5


@dataclass
class CMCLConfig:
    lr: float = 5e-4
    reg: float = 1e-4
    iters: int = 60
    batch_size: int = 128
    temperature: float = 0.05
    self_normalize: bool = True
    topk_focal_k: int = 0
    topk_focal_gamma: float = 1.0
    max_positives: int = 0
    pi_floor: float = 0.01
    max_weight: float = 10.0
    hard_negatives: CMCLHardNegativesConfig = field(default_factory=CMCLHardNegativesConfig)


def _load_positive_map(path: str | Path) -> Dict[str, List[str]]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Positive map not found at {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in {resolved}, found {type(raw).__name__}")
    positives: Dict[str, List[str]] = {}
    for key, value in raw.items():
        if not isinstance(value, list):
            raise ValueError(f"Positive list for key '{key}' must be a list.")
        positives[str(key)] = [str(item) for item in value]
    return positives


def _validate_positive_map(
    positives: Dict[str, List[str]],
    warm_ids: Sequence[str],
    include_self: bool,
) -> Dict[str, List[str]]:
    warm_set = {str(item) for item in warm_ids}
    filtered: Dict[str, List[str]] = {}
    leakage: Dict[str, List[str]] = {}
    for anchor, neighbors in positives.items():
        anchor_str = str(anchor)
        if anchor_str not in warm_set:
            continue
        valid_neighbors: List[str] = []
        leaked: List[str] = []
        for neighbor in neighbors:
            neighbor_str = str(neighbor)
            if neighbor_str in warm_set:
                if neighbor_str not in valid_neighbors:
                    valid_neighbors.append(neighbor_str)
            else:
                leaked.append(neighbor_str)
        if leaked:
            leakage[anchor_str] = leaked
        filtered[anchor_str] = valid_neighbors
    if leakage:
        sample_anchor, leaked_ids = next(iter(leakage.items()))
        raise ValueError(
            f"Positive map leakage detected (e.g. {sample_anchor} -> {leaked_ids[:5]}). "
            "Ensure only warm item ids are present."
        )
    coverage = sum(1 for anchor in warm_ids if str(anchor) in positives) / max(len(warm_ids), 1)
    if coverage < 0.95:
        raise ValueError(f"Positive map coverage below 95% (currently {coverage * 100:.2f}%).")
    for anchor in warm_ids:
        anchor_key = str(anchor)
        neighbors = filtered.setdefault(anchor_key, [])
        if include_self and anchor_key not in neighbors:
            neighbors.insert(0, anchor_key)
    return filtered


class _MICMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        item_ids: Sequence[str],
        warm_features: Sequence[Sequence[float]],
        warm_factors: Sequence[Sequence[float]],
        positives: Dict[str, List[str]] | None,
    ) -> None:
        self.item_ids = [str(item) for item in item_ids]
        self.features = torch.tensor(warm_features, dtype=torch.float32)
        self.factors = torch.tensor(warm_factors, dtype=torch.float32)
        if self.features.shape[0] != len(self.item_ids) or self.factors.shape[0] != len(self.item_ids):
            raise ValueError("Warm features and factors must align with warm item ids.")
        self.positives = positives or {}

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Optional[List[str]]]:
        item_id = self.item_ids[idx]
        pos = self.positives.get(item_id)
        positives_list = list(pos) if pos is not None else []
        return item_id, self.features[idx], self.factors[idx], positives_list


def _micm_collate_fn(
    batch: Sequence[Tuple[str, torch.Tensor, torch.Tensor, Sequence[str]]]
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
    item_ids = [item_id for item_id, _, _, _ in batch]
    features = torch.stack([feat for _, feat, _, _ in batch], dim=0)
    factors = torch.stack([factor for _, _, factor, _ in batch], dim=0)
    positives = [list(pos_list) for _, _, _, pos_list in batch]
    return item_ids, features, factors, positives


def _build_batch_positive_indices(
    batch_item_ids: Sequence[str],
    batch_pos_lists: Sequence[Optional[Sequence[str]]],
) -> List[List[int]]:
    lookup = {item_id: idx for idx, item_id in enumerate(batch_item_ids)}
    indices: List[List[int]] = []
    for row_idx, anchor in enumerate(batch_item_ids):
        candidates = batch_pos_lists[row_idx] or []
        seen: set[str] = set()
        resolved: List[int] = []
        for candidate in candidates:
            candidate_id = str(candidate)
            if candidate_id in lookup and candidate_id not in seen:
                seen.add(candidate_id)
                resolved.append(lookup[candidate_id])
        if not resolved:
            resolved = [row_idx]
        indices.append(resolved)
    return indices


def _info_nce_loss(
    projected: torch.Tensor, targets: torch.Tensor, temperature: float, symmetric: bool
) -> torch.Tensor:
    projected = torch.nn.functional.normalize(projected, dim=1)
    targets = torch.nn.functional.normalize(targets, dim=1)
    logits = projected @ targets.T / max(temperature, 1e-6)
    labels = torch.arange(projected.size(0), device=projected.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    if symmetric:
        logits_rev = targets @ projected.T / max(temperature, 1e-6)
        loss = 0.5 * (
            loss + torch.nn.functional.cross_entropy(logits_rev, labels)
        )
    return loss


def _multi_positive_nce_loss(
    projected: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    positive_indices: Sequence[Sequence[int]],
) -> torch.Tensor:
    projected = torch.nn.functional.normalize(projected, dim=1)
    targets = torch.nn.functional.normalize(targets, dim=1)
    logits = projected @ targets.T / max(temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    denom = exp_logits.sum(dim=1).clamp_min(1e-12)
    batch = projected.size(0)
    pos_mask = torch.zeros((batch, batch), dtype=torch.bool, device=logits.device)
    for row_idx, cols in enumerate(positive_indices):
        if cols:
            pos_mask[row_idx, cols] = True
    pos_sum = (exp_logits * pos_mask).sum(dim=1)
    diag_exp = exp_logits.diagonal()
    pos_sum = torch.where(pos_mask.any(dim=1), pos_sum, diag_exp)
    loss = -torch.log((pos_sum / denom).clamp_min(1e-12))
    return loss.mean()


def _supervised_contrastive_loss(
    projected: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    positive_indices: Sequence[Sequence[int]],
) -> torch.Tensor:
    projected = torch.nn.functional.normalize(projected, dim=1)
    targets = torch.nn.functional.normalize(targets, dim=1)
    logits = projected @ targets.T / max(temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values
    batch = projected.size(0)
    denom_mask = torch.ones((batch, batch), dtype=torch.bool, device=logits.device)
    denom_mask.fill_diagonal_(False)
    masked_logits = logits.masked_fill(~denom_mask, float("-inf"))
    log_denom = torch.logsumexp(masked_logits, dim=1)
    losses: List[torch.Tensor] = []
    for row_idx, cols in enumerate(positive_indices):
        cols = list(cols)
        if not cols:
            cols = [row_idx]
        index_tensor = torch.tensor(cols, dtype=torch.long, device=logits.device)
        pos_logits = logits[row_idx].index_select(0, index_tensor)
        denom_value = log_denom[row_idx]
        if not torch.isfinite(denom_value):
            denom_value = logits[row_idx, row_idx]
        losses.append(-(pos_logits - denom_value).mean())
    return torch.stack(losses).mean()


def train_micm(
    warm_item_ids: Sequence[str],
    warm_features: List[List[float]],
    warm_factors: List[List[float]],
    config: MICMConfig,
    prefer_gpu: bool = True,
) -> MICMModel:
    device = _torch_device(prefer_gpu)
    if not warm_item_ids:
        raise ValueError("warm_item_ids cannot be empty.")
    if len(warm_item_ids) != len(warm_features) or len(warm_item_ids) != len(warm_factors):
        raise ValueError("Warm item ids, features, and factors must share the same length.")

    loss_type = (config.loss_type or "info_nce").lower()
    valid_losses = {"info_nce", "multipos_nce", "supcon"}
    if loss_type not in valid_losses:
        raise ValueError(f"Unsupported MICM loss_type='{config.loss_type}'. Expected one of {sorted(valid_losses)}.")

    print(
        f"[micm] training with loss='{loss_type}' "
        f"(temperature={config.temperature:.4f}, symmetric={config.symmetric})"
    )

    positives_map: Dict[str, List[str]] | None = None
    if config.positives.path:
        try:
            raw_map = _load_positive_map(config.positives.path)
        except FileNotFoundError:
            print(
                f"[micm] positives file '{config.positives.path}' not found; using self-only positives."
            )
        else:
            positives_map = _validate_positive_map(
                raw_map, warm_item_ids, include_self=config.positives.include_self
            )
            covered = sum(1 for key in warm_item_ids if positives_map.get(str(key)))
            avg_len = (
                sum(len(values) for values in positives_map.values()) / max(len(positives_map), 1)
            )
            print(
                f"[micm] loaded positives for {covered}/{len(warm_item_ids)} items; "
                f"avg list length={avg_len:.2f}"
            )
            id_to_idx = {str(item_id): idx for idx, item_id in enumerate(warm_item_ids)}
            normed_factors = torch.nn.functional.normalize(
                torch.tensor(warm_factors, dtype=torch.float32), dim=1
            )
            available = [str(item_id) for item_id in warm_item_ids if positives_map.get(str(item_id))]
            if available:
                rng = random.Random(0)
                for anchor in rng.sample(available, min(3, len(available))):
                    neighbors = positives_map.get(anchor, [])[: max(1, config.positives.k)]
                    anchor_idx = id_to_idx[anchor]
                    anchor_vec = normed_factors[anchor_idx]
                    scored = []
                    for neighbor in neighbors:
                        n_idx = id_to_idx.get(neighbor)
                        if n_idx is None:
                            continue
                        score = float(torch.dot(anchor_vec, normed_factors[n_idx]).item())
                        scored.append((neighbor, score))
                    formatted = ", ".join(f"{nid}:{score:.3f}" for nid, score in scored)
                    print(f"[micm] positives sample {anchor}: {formatted}")
    else:
        if loss_type != "info_nce":
            print("[micm] warning: multi-positive loss selected without positives map; falling back to self-only positives.")

    dataset = _MICMDataset(warm_item_ids, warm_features, warm_factors, positives_map)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=max(0, int(config.num_workers)),
        drop_last=False,
        collate_fn=_micm_collate_fn,
    )

    model = MICMModel(len(warm_features[0]), len(warm_factors[0])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.reg)

    for epoch in range(config.iters):
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_positive_total = 0
        for batch_item_ids, batch_x, batch_y, batch_pos in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            positive_indices = _build_batch_positive_indices(batch_item_ids, batch_pos)

            optimizer.zero_grad()
            projected = model(batch_x)
            if loss_type == "multipos_nce":
                loss = _multi_positive_nce_loss(projected, batch_y, config.temperature, positive_indices)
            elif loss_type == "supcon":
                loss = _supervised_contrastive_loss(projected, batch_y, config.temperature, positive_indices)
            else:
                loss = _info_nce_loss(projected, batch_y, config.temperature, config.symmetric)
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_x.size(0)
            epoch_loss += loss.item() * batch_size_actual
            epoch_samples += batch_size_actual
            epoch_positive_total += sum(len(indices) for indices in positive_indices)

        if epoch_samples > 0:
            avg_loss = epoch_loss / epoch_samples
            avg_pos = epoch_positive_total / epoch_samples
            print(
                f"[micm] epoch {epoch + 1}/{config.iters}: loss={avg_loss:.4f}, "
                f"avg_pos_per_anchor={avg_pos:.2f}"
            )
    return model.cpu()


def infer_micm(
    model: MICMModel,
    cold_features: List[List[float]],
    prefer_gpu: bool = True,
    batch_size: int = 4096,
) -> List[List[float]]:
    device = _torch_device(prefer_gpu)
    model = model.to(device)
    model.eval()
    outputs: List[List[float]] = []
    with torch.no_grad():
        for start in range(0, len(cold_features), batch_size):
            batch = torch.tensor(
                cold_features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            preds = model(batch)
            outputs.extend(preds.cpu().numpy().tolist())
    model.cpu()
    return outputs


class CMCLModel(nn.Module):
    def __init__(self, n_features: int, n_factors: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, n_factors, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _CMCLUserDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        user_ids: Sequence[str],
        user_factors: Sequence[Sequence[float]],
        user_items: Sequence[Sequence[str]],
        pi_lookup: Dict[str, float],
        max_positives: int,
        pi_floor: float,
        max_weight: float,
    ) -> None:
        self.user_ids = list(user_ids)
        self.user_factors = torch.tensor(user_factors, dtype=torch.float32)
        self.user_items = [list(items) for items in user_items]
        self.pi_lookup = pi_lookup
        self.max_positives = max(0, int(max_positives))
        self.pi_floor = max(pi_floor, 1e-4)
        self.max_weight = max(1.0, max_weight)

    def __len__(self) -> int:
        return len(self.user_ids)

    def _sample_items(self, items: List[str], user_id: str) -> List[str]:
        if self.max_positives <= 0 or len(items) <= self.max_positives:
            return items
        rng = random.Random(hash(user_id) & 0xFFFFFFFF)
        return rng.sample(items, self.max_positives)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, List[str], torch.Tensor]:
        user_id = self.user_ids[idx]
        positives = self._sample_items(self.user_items[idx], user_id)
        weights: List[float] = []
        for item in positives:
            key = f"{user_id}::{item}"
            pi = self.pi_lookup.get(key, self.pi_floor)
            pi = max(pi, self.pi_floor)
            inv = min(1.0 / pi, self.max_weight)
            inv = max(1.0, inv)
            weights.append(inv)
        if not weights:
            weights = [1.0]
            positives = [self.user_items[idx][0]]
        return user_id, self.user_factors[idx], positives, torch.tensor(weights, dtype=torch.float32)


def _cmcl_collate_fn(
    feature_tensor: torch.Tensor,
    item_ids: Sequence[str],
    semantic_neighbors: Dict[str, List[str]] | None,
    hard_neg_k: int,
):
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    def _collate(
        batch: Sequence[Tuple[str, torch.Tensor, List[str], torch.Tensor]]
    ) -> Tuple[
        List[str],
        torch.Tensor,
        List[str],
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        user_ids = [entry[0] for entry in batch]
        user_vecs = torch.stack([entry[1] for entry in batch], dim=0)
        positive_items = [entry[2] for entry in batch]
        positive_weights = [entry[3] for entry in batch]
        batch_items: set[str] = set()
        for items in positive_items:
            batch_items.update(items)
        if semantic_neighbors and hard_neg_k > 0:
            for items in positive_items:
                for item in items:
                    for neighbor in semantic_neighbors.get(item, [])[:hard_neg_k]:
                        batch_items.add(neighbor)
        ordered = [item for item in sorted(batch_items) if item in item_index]
        if not ordered:
            ordered = [sorted(item_index.keys())[0]]
        indices = torch.tensor([item_index[item] for item in ordered], dtype=torch.long)
        item_features = feature_tensor.index_select(0, indices)
        lookup = {item: idx for idx, item in enumerate(ordered)}

        positive_indices: List[torch.Tensor] = []
        remapped_weights: List[torch.Tensor] = []
        for items, weights in zip(positive_items, positive_weights):
            idxs: List[int] = []
            vals: List[float] = []
            for item, weight in zip(items, weights.tolist()):
                pos = lookup.get(item)
                if pos is None:
                    continue
                idxs.append(pos)
                vals.append(weight)
            if not idxs:
                idxs = [lookup[ordered[0]]]
                vals = [1.0]
            positive_indices.append(torch.tensor(idxs, dtype=torch.long))
            remapped_weights.append(torch.tensor(vals, dtype=torch.float32))

        return user_ids, user_vecs, ordered, item_features, positive_indices, remapped_weights

    return _collate


def _cmcl_loss(
    logits: torch.Tensor,
    positive_indices: Sequence[torch.Tensor],
    positive_weights: Sequence[torch.Tensor],
    temperature: float,
    self_normalize: bool,
    topk_focal_k: int,
    topk_focal_gamma: float,
) -> torch.Tensor:
    log_denom = torch.logsumexp(logits, dim=1)
    total = logits.new_tensor(0.0)
    total_weight = logits.new_tensor(0.0)
    for row_idx, cols in enumerate(positive_indices):
        if cols.numel() == 0:
            continue
        weights = positive_weights[row_idx].to(logits.device)
        cols = cols.to(logits.device)
        if weights.numel() != cols.numel():
            continue
        if self_normalize:
            weights = weights / weights.sum().clamp_min(1e-6)
        pos_logits = logits[row_idx].index_select(0, cols)
        if topk_focal_k > 0 and pos_logits.numel() > 0:
            sorted_vals, _ = torch.sort(pos_logits, descending=True)
            kth_idx = min(topk_focal_k - 1, sorted_vals.numel() - 1)
            kth_score = sorted_vals[kth_idx]
            margins = torch.relu(kth_score - pos_logits)
            focal = 1.0 + topk_focal_gamma * torch.exp(-margins / max(temperature, 1e-6))
            weights = weights * focal
            if self_normalize:
                weights = weights / weights.sum().clamp_min(1e-6)
        total += -(weights * (pos_logits - log_denom[row_idx])).sum()
        total_weight += weights.sum()
    return total / total_weight.clamp_min(1e-6)


def _build_semantic_neighbors(
    warm_features: List[List[float]],
    warm_item_ids: Sequence[str],
    k: int,
    min_sim: float,
) -> Dict[str, List[str]]:
    if k <= 0:
        return {}
    features = torch.tensor(warm_features, dtype=torch.float32)
    features = torch.nn.functional.normalize(features, dim=1)
    n_items = features.size(0)
    neighbors: Dict[str, List[str]] = {}
    block = 512
    for start in range(0, n_items, block):
        end = min(n_items, start + block)
        sims = features[start:end] @ features.T
        for row in range(end - start):
            item_idx = start + row
            sims_row = sims[row]
            sims_row[item_idx] = -1.0
            topk = torch.topk(sims_row, k=min(k, n_items - 1))
            selected: List[str] = []
            for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
                if score < min_sim:
                    continue
                selected.append(warm_item_ids[idx])
            neighbors[warm_item_ids[item_idx]] = selected
    return neighbors


def train_cmcl(
    warm_rows: Sequence[dict],
    warm_item_ids: Sequence[str],
    warm_features: List[List[float]],
    user_factors: List[List[float]],
    user_to_idx: Dict[str, int],
    pi_lookup: Dict[str, float],
    config: CMCLConfig,
    prefer_gpu: bool = True,
) -> CMCLModel:
    device = _torch_device(prefer_gpu)
    feature_tensor = torch.tensor(warm_features, dtype=torch.float32)
    idx_to_user = [""] * len(user_to_idx)
    for user, idx in user_to_idx.items():
        idx_to_user[idx] = user
    user_items: List[List[str]] = [[] for _ in idx_to_user]
    warm_item_set = set(warm_item_ids)
    for row in warm_rows:
        user = row["user_id"]
        item = row["item_id"]
        if item not in warm_item_set:
            continue
        user_idx = user_to_idx.get(user)
        if user_idx is None:
            continue
        if item not in user_items[user_idx]:
            user_items[user_idx].append(item)
    filtered_ids: List[str] = []
    filtered_factors: List[List[float]] = []
    filtered_items: List[List[str]] = []
    for idx, items in enumerate(user_items):
        if not items:
            continue
        filtered_ids.append(idx_to_user[idx])
        filtered_factors.append(user_factors[idx])
        filtered_items.append(items)
    if not filtered_ids:
        raise RuntimeError("No eligible users found for CMCL training.")
    dataset = _CMCLUserDataset(
        filtered_ids,
        filtered_factors,
        filtered_items,
        pi_lookup,
        config.max_positives,
        config.pi_floor,
        config.max_weight,
    )
    semantic_neighbors = None
    if config.hard_negatives.k > 0:
        print(
            f"[cmcl] mining semantic negatives (k={config.hard_negatives.k}, "
            f"min_sim={config.hard_negatives.min_sim})"
        )
        semantic_neighbors = _build_semantic_neighbors(
            warm_features, warm_item_ids, config.hard_negatives.k, config.hard_negatives.min_sim
        )
    collate = _cmcl_collate_fn(
        feature_tensor,
        warm_item_ids,
        semantic_neighbors,
        config.hard_negatives.k,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, config.batch_size),
        shuffle=True,
        drop_last=False,
        collate_fn=collate,
    )
    model = CMCLModel(len(warm_features[0]), len(user_factors[0])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.reg)

    for epoch in range(config.iters):
        epoch_loss = 0.0
        batches = 0
        for (
            _,
            user_vecs,
            _,
            item_features,
            pos_indices,
            pos_weights,
        ) in loader:
            user_vecs = user_vecs.to(device)
            item_features = item_features.to(device)
            projected = model(item_features)
            logits = user_vecs @ projected.T
            logits = logits / max(config.temperature, 1e-6)
            loss = _cmcl_loss(
                logits,
                pos_indices,
                pos_weights,
                config.temperature,
                config.self_normalize,
                config.topk_focal_k,
                config.topk_focal_gamma,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        if batches:
            print(f"[cmcl] epoch {epoch + 1}/{config.iters}: loss={epoch_loss / batches:.4f}")
    return model.cpu()


def infer_cmcl(
    model: CMCLModel,
    features: List[List[float]],
    prefer_gpu: bool = True,
    batch_size: int = 4096,
) -> List[List[float]]:
    device = _torch_device(prefer_gpu)
    model = model.to(device)
    model.eval()
    outputs: List[List[float]] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            preds = model(batch)
            outputs.extend(preds.cpu().numpy().tolist())
    model.cpu()
    return outputs


class CDLModel(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_factors: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden_dim), nn.ReLU())
        self.decoder = nn.Linear(hidden_dim, n_features)
        self.factor_head = nn.Linear(hidden_dim, n_factors)

    def forward(self, x: torch.Tensor, corruption: float = 0.0, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if training and corruption > 0.0:
            noise = torch.bernoulli(torch.ones_like(x) * (1.0 - corruption))
            x = x * noise
        hidden = self.encoder(x)
        recon = self.decoder(hidden)
        factors = self.factor_head(hidden)
        return recon, factors


@dataclass
class CDLTorchConfig:
    hidden_dim: int = 256
    lr: float = 5e-3
    reg: float = 1e-4
    iters: int = 300
    batch_size: int = 256
    corruption: float = 0.2
    factor_weight: float = 1.0


def train_cdl(
    warm_features: List[List[float]],
    warm_factors: List[List[float]],
    config: CDLTorchConfig,
    prefer_gpu: bool = True,
) -> CDLModel:
    device = _torch_device(prefer_gpu)
    model = CDLModel(len(warm_features[0]), config.hidden_dim, len(warm_factors[0])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.reg)

    X = _tensor_from(warm_features, device)
    Y = _tensor_from(warm_factors, device)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for _ in range(config.iters):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            recon, factors = model(batch_x, corruption=config.corruption, training=True)
            recon_loss = torch.nn.functional.mse_loss(recon, batch_x)
            factor_loss = torch.nn.functional.mse_loss(factors, batch_y)
            loss = recon_loss + config.factor_weight * factor_loss
            loss.backward()
            optimizer.step()
    return model.cpu()


def infer_cdl(
    model: CDLModel,
    features: List[List[float]],
    prefer_gpu: bool = True,
    batch_size: int = 4096,
) -> List[List[float]]:
    device = _torch_device(prefer_gpu)
    model = model.to(device)
    model.eval()
    outputs: List[List[float]] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            _, factors = model(batch)
            outputs.extend(factors.cpu().numpy().tolist())
    model.cpu()
    return outputs


def train_ctpf(
    warm_features: List[List[float]],
    warm_factors: List[List[float]],
    topics: int,
    nmf_iters: int,
    reg: float,
    prefer_gpu: bool,
    seed: int,
) -> Dict[str, torch.Tensor]:
    device = _torch_device(prefer_gpu)
    _set_seed(seed)
    X = _tensor_from(warm_features, device)
    eps = 1e-8
    n_items, n_features = X.shape
    W = torch.rand((n_items, topics), device=device) + eps
    H = torch.rand((topics, n_features), device=device) + eps

    for _ in range(nmf_iters):
        WH = W @ H + eps
        ratio = torch.matmul(X / WH, H.transpose(1, 0))
        W = W * ratio
        W = W.clamp_min(eps)
        W = W / W.sum(dim=1, keepdim=True).clamp_min(eps)
        WH = W @ H + eps
        H = H * (W.T @ (X / WH))
        H = H.clamp_min(eps)

    V = _tensor_from(warm_factors, device)
    XtX = W.T @ W + reg * torch.eye(topics, device=device)
    topic_to_factor = torch.linalg.solve(XtX, W.T @ V)

    return {"topic_components": H.cpu(), "topic_to_factor": topic_to_factor.cpu()}


def project_topics(
    features: List[List[float]],
    topic_components: torch.Tensor,
    iters: int,
    seed: int,
    prefer_gpu: bool,
) -> torch.Tensor:
    device = _torch_device(prefer_gpu)
    _set_seed(seed)
    X = _tensor_from(features, device)
    H = topic_components.to(device)
    eps = 1e-8
    n_items = X.shape[0]
    topics = H.shape[0]
    W = torch.rand((n_items, topics), device=device) + eps
    HT = H.transpose(1, 0)

    for _ in range(iters):
        WH = W @ H
        numerator = torch.matmul(X, HT)
        denominator = torch.matmul(WH, HT) + eps
        W = W * (numerator / denominator.clamp_min(eps))
        W = W.clamp_min(eps)
        W = W / W.sum(dim=1, keepdim=True).clamp_min(eps)
    return W


def infer_ctpf(
    cold_features: List[List[float]],
    params: Dict[str, torch.Tensor],
    projection_iters: int,
    seed: int,
    prefer_gpu: bool,
) -> List[List[float]]:
    W = project_topics(cold_features, params["topic_components"], projection_iters, seed, prefer_gpu)
    topic_to_factor = params["topic_to_factor"].to(W.device)
    if topic_to_factor.shape[0] != W.shape[1]:
        topic_to_factor = topic_to_factor.T
    factors = W @ topic_to_factor
    return factors.cpu().numpy().tolist()


def train_hft(
    warm_features: List[List[float]],
    warm_factors: List[List[float]],
    topics: int,
    nmf_iters: int,
    kappa: float,
    reg: float,
    prefer_gpu: bool,
    seed: int,
) -> Dict[str, torch.Tensor]:
    params = train_ctpf(
        warm_features,
        warm_factors,
        topics=topics,
        nmf_iters=nmf_iters,
        reg=reg,
        prefer_gpu=prefer_gpu,
        seed=seed,
    )
    device = _torch_device(prefer_gpu)
    V = _tensor_from(warm_factors, device)
    scaled = torch.tanh(kappa * V)
    W_topics = project_topics(warm_features, params["topic_components"], nmf_iters, seed, prefer_gpu).to(device)
    XtX = W_topics.T @ W_topics + reg * torch.eye(W_topics.shape[1], device=device)
    topic_to_scaled = torch.linalg.solve(XtX, W_topics.T @ scaled)
    return {
        "topic_components": params["topic_components"],
        "topic_to_scaled": topic_to_scaled.cpu(),
        "kappa": kappa,
    }


def infer_hft(
    cold_features: List[List[float]],
    params: Dict[str, torch.Tensor],
    projection_iters: int,
    seed: int,
    prefer_gpu: bool,
) -> List[List[float]]:
    W = project_topics(cold_features, params["topic_components"], projection_iters, seed, prefer_gpu)
    topic_to_scaled = params["topic_to_scaled"].to(W.device)
    scaled = W @ topic_to_scaled
    kappa = params["kappa"]
    clipped = scaled.clamp(-0.999, 0.999)
    factors = 0.5 * torch.log((1 + clipped) / (1 - clipped + 1e-8)) / max(kappa, 1e-6)
    return factors.cpu().numpy().tolist()


__all__ = [
    "train_mf",
    "score_users_on_cold",
    "CtrliteTorchConfig",
    "train_ctrlite",
    "infer_ctrlite",
    "A2FTorchConfig",
    "train_a2f",
    "infer_a2f",
    "MICMConfig",
    "train_micm",
    "infer_micm",
    "CMCLConfig",
    "train_cmcl",
    "infer_cmcl",
    "CDLTorchConfig",
    "train_cdl",
    "infer_cdl",
    "train_ctpf",
    "infer_ctpf",
    "train_hft",
    "infer_hft",
]
