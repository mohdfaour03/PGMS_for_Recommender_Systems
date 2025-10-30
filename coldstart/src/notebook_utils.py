"""Helpers for running the cold-start workflow inside notebooks.

This module packages the lightweight YAML reader and MovieLens fetcher that
were previously embedded in the command-line scripts so that notebook users
do not need to depend on the terminal entrypoints.
"""
from __future__ import annotations

import ast
import io
import zipfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests


MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_MEDIUM_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"


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

    movies["item_text"] = (
        movies["title"].fillna("")
        + " "
        + movies["genres"].fillna("").str.replace("|", " ", regex=False)
    ).str.strip()

    frame = ratings.merge(movies[["movieId", "item_text"]], on="movieId", how="left")
    frame.rename(
        columns={
            "userId": "user_id",
            "movieId": "item_id",
            "rating": "rating_or_y",
        },
        inplace=True,
    )
    frame["item_text"] = frame["item_text"].fillna("")

    if limit is not None and limit > 0:
        frame = frame.head(limit)

    return frame[["user_id", "item_id", "rating_or_y", "item_text"]]


__all__ = [
    "_read_simple_yaml",
    "build_interaction_frame",
    "MOVIELENS_SMALL_URL",
    "MOVIELENS_MEDIUM_URL",
]
