"""Lightweight TF-IDF featurizer with leakage guard."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple


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
