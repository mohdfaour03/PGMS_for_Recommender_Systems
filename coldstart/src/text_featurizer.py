"""Lightweight text featurizers with leakage guard."""
from __future__ import annotations

import math
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

    def fit_warm(self, item_texts: Sequence[str]) -> None:
        print("TF-IDF fit on warm only")
        self.vectorizer.fit(item_texts)

    def transform(self, item_texts: Sequence[str]) -> List[List[float]]:
        return self.vectorizer.transform(item_texts)

    def fit_transform_warm(self, item_texts: Sequence[str]) -> List[List[float]]:
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
        max_length: int = 64,
        pooling: str = "cls",
        proj_dim: int | None = 256,
        device: str | None = None,
    ) -> None:
        if torch is None or AutoModel is None or AutoTokenizer is None:  # pragma: no cover - dynamic import
            raise ImportError("transformers and torch are required for frozen_bert_linear.")
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.max_length = max(8, max_length)
        self.pooling = pooling if pooling in {"cls", "mean"} else "cls"
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

    def _embed(self, item_texts: Sequence[str]) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for start in range(0, len(item_texts), self.batch_size):
            batch = [text or "" for text in item_texts[start : start + self.batch_size]]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.no_grad():
                result = self._model(**encoded)
                hidden = result.last_hidden_state
                if self.pooling == "mean":
                    pooled = hidden.mean(dim=1)
                else:
                    pooled = hidden[:, 0, :]
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

    def fit_warm(self, item_texts: Sequence[str]) -> None:
        embeddings = self._embed(item_texts)
        _ = self._project_embeddings(embeddings, fit=True)

    def transform(self, item_texts: Sequence[str]) -> List[List[float]]:
        embeddings = self._embed(item_texts)
        projected = self._project_embeddings(embeddings, fit=False)
        return projected.numpy().tolist()

    def fit_transform_warm(self, item_texts: Sequence[str]) -> List[List[float]]:
        embeddings = self._embed(item_texts)
        projected = self._project_embeddings(embeddings, fit=True)
        return projected.numpy().tolist()

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
