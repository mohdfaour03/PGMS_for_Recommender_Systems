"""PyTorch-backed training utilities with optional GPU acceleration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

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
        ratio = (X / WH).matmul(H.transpose(1, 0))
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
    HT = H.T

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
    "CDLTorchConfig",
    "train_cdl",
    "infer_cdl",
    "train_ctpf",
    "infer_ctpf",
    "train_hft",
    "infer_hft",
]
