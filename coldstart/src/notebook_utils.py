"""Helpers for running the cold-start workflow inside notebooks.

This module packages the lightweight YAML reader and MovieLens fetcher that
were previously embedded in the command-line scripts so that notebook users
do not need to depend on the terminal entrypoints.
"""
from __future__ import annotations

import ast
import io
import zipfile
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests


MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_MEDIUM_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
_YEAR_SUFFIX = re.compile(r"\s*\((\d{4})\)\s*$")
_MULTISPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _strip_year_suffix(title: str) -> tuple[str, int | None]:
    if not isinstance(title, str):
        return "", None
    match = _YEAR_SUFFIX.search(title)
    year = int(match.group(1)) if match else None
    return _YEAR_SUFFIX.sub("", title).strip(), year


def _normalize_token_block(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lowered = text.lower()
    lowered = _NON_ALNUM.sub(" ", lowered)
    return _MULTISPACE.sub(" ", lowered).strip()


def _normalize_item_text(title: str, genres: str, tags: str) -> str:
    parts = [title, genres, tags]
    combined = " ".join(part for part in parts if part)
    return _MULTISPACE.sub(" ", combined.lower()).strip()


def _aggregate_tags(tags: pd.DataFrame) -> Dict[int, str]:
    if tags.empty or "tag" not in tags.columns:
        return {}
    tags["tag"] = tags["tag"].fillna("").astype(str)
    tags["tag"] = tags["tag"].map(_normalize_token_block)
    tags = tags[tags["tag"].str.len() > 0]
    if tags.empty:
        return {}

    def _join(values: pd.Series) -> str:
        seen = dict.fromkeys(values.tolist())
        return " ".join(seen.keys())

    grouped = tags.groupby("movieId")["tag"].agg(_join)
    return grouped.to_dict()


def _year_to_timestamp(year: int | None) -> int | None:
    if year is None or year <= 0:
        return None
    dt = datetime(year, 1, 1, tzinfo=timezone.utc)
    return int(dt.timestamp())


def _read_simple_yaml(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Parse the minimal YAML subset used by the project configs."""
    content = Path(path).read_text(encoding="utf-8").splitlines()
    config: Dict[str, Dict[str, Any]] = {}
    current = None
    for line in content:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith(":"):
            current = stripped[:-1]
            config[current] = {}
        elif ":" in stripped and current is not None:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                parsed = value
            config[current][key] = parsed
    return config


def build_interaction_frame(dataset: str = "medium", limit: int | None = None) -> pd.DataFrame:
    """Download a MovieLens dataset and return the schema-compatible frame.

    Args:
        dataset: One of ``{"small", "medium"}``. ``small`` maps to ``ml-latest-small``
            (100k ratings). ``medium`` (the default) maps to ``ml-latest`` (roughly 1M ratings).
        limit: Optional cap on the number of rows for quick smoke tests.
    """
    dataset = dataset.lower()
    if dataset not in {"small", "medium"}:
        raise ValueError("dataset must be either 'small' or 'medium'")

    url = MOVIELENS_SMALL_URL if dataset == "small" else MOVIELENS_MEDIUM_URL
    archive_prefix = "ml-latest-small" if dataset == "small" else "ml-latest"

    response = requests.get(url, timeout=120 if dataset == "medium" else 60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open(f"{archive_prefix}/ratings.csv") as fh:
            ratings = pd.read_csv(fh)
        with archive.open(f"{archive_prefix}/movies.csv") as fh:
            movies = pd.read_csv(fh)
        with archive.open(f"{archive_prefix}/tags.csv") as fh:
            tags = pd.read_csv(fh)

    tag_map = _aggregate_tags(tags)
    movies["title"] = movies["title"].fillna("")
    stripped = movies["title"].apply(_strip_year_suffix)
    movies["title_no_year"] = stripped.apply(lambda pair: pair[0])
    movies["release_year"] = stripped.apply(lambda pair: pair[1]).fillna(-1).astype(int)
    movies["release_ts"] = movies["release_year"].apply(
        lambda year: _year_to_timestamp(year if year > 0 else None)
    )
    movies["release_ts"] = movies["release_ts"].fillna(-1).astype(int)
    movies["item_genres"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    movies["item_tags"] = movies["movieId"].map(tag_map).fillna("")
    movies["item_text"] = movies.apply(
        lambda row: _normalize_item_text(row["title_no_year"], row["item_genres"], row["item_tags"]),
        axis=1,
    )
    movies["text_len"] = movies["item_text"].str.split().apply(len).astype(int)

    frame = ratings.merge(
        movies[
            [
                "movieId",
                "item_text",
                "item_genres",
                "item_tags",
                "release_year",
                "release_ts",
                "text_len",
            ]
        ],
        on="movieId",
        how="left",
    )
    frame.rename(
        columns={
            "userId": "user_id",
            "movieId": "item_id",
            "rating": "rating_or_y",
            "timestamp": "timestamp",
        },
        inplace=True,
    )
    frame["item_text"] = frame["item_text"].fillna("")
    frame["item_genres"] = frame["item_genres"].fillna("")
    frame["item_tags"] = frame["item_tags"].fillna("")
    frame["release_year"] = frame["release_year"].fillna(-1).astype(int)
    frame["release_ts"] = frame["release_ts"].fillna(-1).astype(int)
    frame["text_len"] = frame["text_len"].fillna(0).astype(int)
    frame["timestamp"] = frame["timestamp"].astype(int)

    if limit is not None and limit > 0:
        frame = frame.head(limit)

    return frame[
        [
            "user_id",
            "item_id",
            "rating_or_y",
            "timestamp",
            "item_text",
            "item_genres",
            "item_tags",
            "release_year",
            "release_ts",
            "text_len",
        ]
    ]


__all__ = [
    "_read_simple_yaml",
    "build_interaction_frame",
    "MOVIELENS_SMALL_URL",
    "MOVIELENS_MEDIUM_URL",
]
