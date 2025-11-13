"""Lightweight text featurizers with leakage guard."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    torch = None
    AutoModel = None
    AutoTokenizer = None


class SimpleTfidfVectorizer:
    """A minimal TF-IDF implementation without external dependencies."""

    def __init__(
        self,
        max_features: int | None = None,
        min_df: float = 1.0,
        ngram_range: Tuple[int, int] = (1, 1),
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vocabulary_: Dict[str, int] | None = None
        self.idf_: List[float] | None = None

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in text.lower().split() if token]

    def _ngrams(self, tokens: Sequence[str]) -> List[str]:
        min_n, max_n = self.ngram_range
        ngrams: List[str] = []
        for n in range(min_n, max_n + 1):
            if n <= 0:
                continue
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])
                ngrams.append(ngram)
        return ngrams

    def fit(self, documents: Sequence[str]) -> "SimpleTfidfVectorizer":
        tokenized = [self._ngrams(self._tokenize(doc)) for doc in documents]
        doc_freq: Dict[str, int] = defaultdict(int)
        for tokens in tokenized:
            for term in set(tokens):
                doc_freq[term] += 1

        n_docs = max(len(documents), 1)
        min_df_threshold = self.min_df
        if 0 < self.min_df < 1:
            min_df_threshold = math.ceil(self.min_df * n_docs)
        terms = [
            term
            for term, freq in doc_freq.items()
            if freq >= int(min_df_threshold)
        ]
        terms.sort(key=lambda term: (-doc_freq[term], term))
        if self.max_features is not None:
            terms = terms[: self.max_features]

        self.vocabulary_ = {term: idx for idx, term in enumerate(terms)}
        if not self.vocabulary_:
            self.idf_ = []
            return self

        idf_values = [0.0] * len(self.vocabulary_)
        for term, idx in self.vocabulary_.items():
            df = doc_freq[term]
            idf_values[idx] = math.log((1 + n_docs) / (1 + df)) + 1.0
        self.idf_ = idf_values
        return self

    def transform(self, documents: Sequence[str]) -> List[List[float]]:
        if self.vocabulary_ is None or self.idf_ is None:
            raise RuntimeError("Vectorizer must be fit before calling transform().")
        n_features = len(self.vocabulary_)
        features: List[List[float]] = []
        for doc in documents:
            tokens = self._ngrams(self._tokenize(doc))
            counts = Counter(tokens)
            if not counts:
                features.append([0.0] * n_features)
                continue
            total = sum(counts.values())
            row = [0.0] * n_features
            for term, count in counts.items():
                col = self.vocabulary_.get(term)
                if col is None:
                    continue
                tf = count / total
                row[col] = tf * self.idf_[col]
            features.append(row)
        return features

    def fit_transform(self, documents: Sequence[str]) -> List[List[float]]:
        self.fit(documents)
        return self.transform(documents)


class TfidfTextFeaturizer:
    """Wrapper around :class:`SimpleTfidfVectorizer` with logging."""

    def __init__(
        self,
        max_features: int | None = None,
        min_df: float = 1.0,
        ngram_range: Tuple[int, int] = (1, 1),
    ) -> None:
        self.vectorizer = SimpleTfidfVectorizer(
            max_features=max_features, min_df=min_df, ngram_range=ngram_range
        )

    def fit_warm(self, item_texts: Sequence[str], **_: object) -> None:
        print("TF-IDF fit on warm only")
        self.vectorizer.fit(item_texts)

    def transform(self, item_texts: Sequence[str], **_: object) -> List[List[float]]:
        return self.vectorizer.transform(item_texts)

    def fit_transform_warm(self, item_texts: Sequence[str], **_: object) -> List[List[float]]:
        print("TF-IDF fit on warm only")
        return self.vectorizer.fit_transform(item_texts)

    def save_state(self) -> dict:
        return {
            "max_features": self.vectorizer.max_features,
            "min_df": self.vectorizer.min_df,
            "ngram_range": self.vectorizer.ngram_range,
            "vocabulary": self.vectorizer.vocabulary_,
            "idf": self.vectorizer.idf_,
        }

    def load_state(self, state: dict) -> None:
        self.vectorizer = SimpleTfidfVectorizer(
            max_features=state.get("max_features"),
            min_df=state.get("min_df", 1.0),
            ngram_range=tuple(state.get("ngram_range", (1, 1))),
        )
        vocab = state.get("vocabulary") or {}
        self.vectorizer.vocabulary_ = {str(k): int(v) for k, v in vocab.items()}
        self.vectorizer.idf_ = [float(x) for x in state.get("idf", [])]


class FrozenBertLinearFeaturizer:
    """Use a frozen BERT encoder plus optional linear projection for item text."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 32,
        max_length: int = 128,
        pooling: str = "mean",
        proj_dim: int | None = 256,
        device: str | None = None,
        log_batch_stats_every: int = 0,
        bucket_dropout: Dict[str, float] | None = None,
    ) -> None:
        if torch is None or AutoModel is None or AutoTokenizer is None:  # pragma: no cover - dynamic import
            raise ImportError("transformers and torch are required for frozen_bert_linear.")
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.max_length = max(8, min(128, max_length))
        valid_pooling = {"cls", "mean", "cls_mean"}
        self.pooling = pooling if pooling in valid_pooling else "mean"
        self.proj_dim = proj_dim
        self.device_name = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False
        if self.device_name:
            device_obj = torch.device(self.device_name)
        else:
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device_obj
        self._model.to(self._device)
        self._mean: torch.Tensor | None = None
        self._proj: torch.Tensor | None = None
        self._last_token_lengths: List[float] = []
        self._last_token_buckets: List[str] = []
        self._log_batch_stats_every = max(0, int(log_batch_stats_every))
        self._bucket_dropout = {
            (key or "").lower(): max(0.0, float(value))
            for key, value in (bucket_dropout or {"medium": 0.1}).items()
            if value and float(value) > 0
        }
        self._token_dropout_rng = random.Random(1234)

    def _log_batch_stats(
        self,
        log_prefix: str,
        batch_index: int,
        token_lengths: torch.Tensor,
        bucket_ids: Sequence[str] | None,
    ) -> None:
        content_lengths = torch.clamp(token_lengths - 2, min=0).to(torch.float32)
        if content_lengths.numel() == 0:
            return
        labels = list(bucket_ids) if bucket_ids is not None else ["all"] * content_lengths.shape[0]
        bucket_to_lengths: dict[str, List[float]] = {}
        for length, label in zip(content_lengths.tolist(), labels):
            bucket_to_lengths.setdefault(label or "unknown", []).append(length)
        for bucket, values in bucket_to_lengths.items():
            mean_len = sum(values) / max(len(values), 1)
            zero_pct = sum(1 for val in values if val <= 0.0) / max(len(values), 1) * 100.0
            print(
                f"[text] {log_prefix} batch={batch_index} bucket={bucket} "
                f"mean_tokens={mean_len:.1f} zero_pct={zero_pct:.2f}% "
                f"max_seq_len={self.max_length}"
            )

    def _maybe_dropout_tokens(self, text: str, bucket_id: str | None) -> str:
        bucket_key = (bucket_id or "").lower()
        prob = self._bucket_dropout.get(bucket_key, 0.0)
        if prob <= 0.0:
            return text
        tokens = text.split()
        if not tokens:
            return text
        kept: List[str] = []
        for token in tokens:
            if self._token_dropout_rng.random() > prob:
                kept.append(token)
        if not kept:
            kept = tokens[:1]
        return " ".join(kept)

    def _embed(
        self,
        item_texts: Sequence[str],
        bucket_ids: Sequence[str] | None = None,
        log_prefix: str = "[text]",
    ) -> torch.Tensor:
        self._last_token_lengths = []
        self._last_token_buckets = []
        outputs: List[torch.Tensor] = []
        all_buckets = list(bucket_ids) if bucket_ids is not None else None
        for start in range(0, len(item_texts), self.batch_size):
            batch = [text or "" for text in item_texts[start : start + self.batch_size]]
            batch_bucket = (
                all_buckets[start : start + self.batch_size]
                if all_buckets is not None
                else [None] * len(batch)
            )
            processed_batch = [
                self._maybe_dropout_tokens(text, bucket_id) for text, bucket_id in zip(batch, batch_bucket)
            ]
            encoded = self._tokenizer(
                processed_batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            token_lengths = encoded["attention_mask"].sum(dim=1)
            self._last_token_lengths.extend(token_lengths.tolist())
            if batch_bucket is not None:
                self._last_token_buckets.extend([(bucket or "unknown") for bucket in batch_bucket])
            with torch.no_grad():
                result = self._model(**encoded)
                hidden = result.last_hidden_state
                if self.pooling == "cls":
                    pooled_cls = hidden[:, 0, :]
                    pooled = pooled_cls
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    mask = mask.to(hidden.dtype)
                    summed = (hidden * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp_min(1.0)
                    pooled_mean = summed / counts
                    if self.pooling == "cls_mean":
                        pooled = torch.cat([pooled_mean, hidden[:, 0, :]], dim=1)
                    else:
                        pooled = pooled_mean
            if self._log_batch_stats_every and (
                (start // self.batch_size) % self._log_batch_stats_every == 0
            ):
                self._log_batch_stats(log_prefix, start // self.batch_size, token_lengths, batch_bucket)
            outputs.append(pooled.cpu())
        if not outputs:
            return torch.zeros((0, self._model.config.hidden_size))
        return torch.cat(outputs, dim=0)

    def _project_embeddings(self, embeddings: torch.Tensor, fit: bool) -> torch.Tensor:
        if embeddings.numel() == 0:
            return embeddings
        if fit:
            self._mean = embeddings.mean(dim=0, keepdim=True)
        if self._mean is None:
            raise RuntimeError("BERT featurizer must be fit before transform().")
        centered = embeddings - self._mean
        if fit and self.proj_dim and 0 < self.proj_dim < centered.shape[1]:
            q = min(centered.shape[0], centered.shape[1], self.proj_dim + 8)
            q = max(q, self.proj_dim)
            U, S, V = torch.pca_lowrank(centered, q=q, center=False)
            self._proj = V[:, : self.proj_dim]
        if self._proj is not None:
            return centered @ self._proj
        return centered

    def fit_warm(
        self,
        item_texts: Sequence[str],
        *,
        bucket_ids: Sequence[str] | None = None,
        log_prefix: str = "[text][warm]",
    ) -> None:
        embeddings = self._embed(item_texts, bucket_ids=bucket_ids, log_prefix=log_prefix)
        _ = self._project_embeddings(embeddings, fit=True)

    def transform(
        self,
        item_texts: Sequence[str],
        *,
        bucket_ids: Sequence[str] | None = None,
        log_prefix: str = "[text][transform]",
    ) -> List[List[float]]:
        embeddings = self._embed(item_texts, bucket_ids=bucket_ids, log_prefix=log_prefix)
        projected = self._project_embeddings(embeddings, fit=False)
        return projected.numpy().tolist()

    def fit_transform_warm(
        self,
        item_texts: Sequence[str],
        *,
        bucket_ids: Sequence[str] | None = None,
        log_prefix: str = "[text][warm_fit_transform]",
    ) -> List[List[float]]:
        embeddings = self._embed(item_texts, bucket_ids=bucket_ids, log_prefix=log_prefix)
        projected = self._project_embeddings(embeddings, fit=True)
        return projected.numpy().tolist()

    def get_last_token_lengths(self) -> List[float]:
        return list(self._last_token_lengths)

    def get_last_token_buckets(self) -> List[str]:
        return list(self._last_token_buckets)

    def save_state(self) -> dict:
        if self._mean is None:
            raise RuntimeError("Cannot save state before fit.")
        state = {
            "type": "frozen_bert_linear",
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "pooling": self.pooling,
            "proj_dim": self.proj_dim,
            "device": str(self._device),
            "mean": self._mean.detach().cpu().numpy().tolist(),
        }
        if self._proj is not None:
            state["proj"] = self._proj.detach().cpu().numpy().tolist()
        return state

    def load_state(self, state: dict) -> None:
        self._mean = torch.tensor(state["mean"], dtype=torch.float32)
        proj = state.get("proj")
        self._proj = torch.tensor(proj, dtype=torch.float32) if proj is not None else None


def build_text_featurizer(kind: str = "tfidf", params: dict | None = None):
    params = params or {}
    key = (kind or "tfidf").strip().lower()
    if key == "tfidf":
        return TfidfTextFeaturizer(**params)
    if key == "frozen_bert_linear":
        return FrozenBertLinearFeaturizer(**params)
    raise ValueError(f"Unknown text encoder '{kind}'.")


__all__ = [
    "SimpleTfidfVectorizer",
    "TfidfTextFeaturizer",
    "FrozenBertLinearFeaturizer",
    "build_text_featurizer",
]
