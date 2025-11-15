"""Helpers for running the cold-start workflow inside notebooks.

This module packages the lightweight YAML reader and MovieLens fetcher that
were previously embedded in the command-line scripts so that notebook users
do not need to depend on the terminal entrypoints.
"""
from __future__ import annotations

import ast
import csv
import io
import zipfile
from datetime import datetime, timezone
import html
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
from requests import exceptions as requests_exceptions
import ssl
import urllib.request
from urllib.parse import quote


MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_MEDIUM_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
_AMAZON_BASE_NAMES = {
    "beauty": "All_Beauty",
    "baby": "Baby",
    "sports": "Sports_and_Outdoors",
    "electronics": "Electronics",
    "home": "Home_and_Kitchen",
    "digital_music": "Digital_Music",
}
AMAZON_DATASETS = sorted(_AMAZON_BASE_NAMES.keys())
_AMAZON_REVIEW_MIRRORS = [
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall",
    "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall",
]
_AMAZON_META_MIRRORS = [
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2",
    "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2",
]
GOODREADS_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
_GOODREADS_GENRE_SLUGS = {
    "children": "children",
    "comics_graphic": "comics_graphic",
    "fantasy_paranormal": "fantasy_paranormal",
    "history_biography": "history_biography",
    "mystery_thriller_crime": "mystery_thriller_crime",
    "poetry": "poetry",
    "romance": "romance",
    "young_adult": "young_adult",
}
GOODREADS_GENRES = sorted(_GOODREADS_GENRE_SLUGS.keys())
_MSNEWS_VARIANTS = {
    "mind_small_train": {
        "url": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
        "folder": "MINDsmall_train",
        "zip_name": "MINDsmall_train.zip",
    },
    "mind_small_dev": {
        "url": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
        "folder": "MINDsmall_dev",
        "zip_name": "MINDsmall_dev.zip",
    },
}
MSNEWS_VARIANTS = sorted(_MSNEWS_VARIANTS.keys())
_MSNEWS_TIMESTAMP_FORMAT = "%m/%d/%Y %I:%M:%S %p"
_YEAR_SUFFIX = re.compile(r"\s*\((\d{4})\)\s*$")
_MULTISPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_HTML_TAG = re.compile(r"<[^>]+>")
_GOODREADS_TS_FORMAT = "%a %b %d %H:%M:%S %z %Y"


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


def _normalize_free_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = html.unescape(text)
    cleaned = _HTML_TAG.sub(" ", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ")
    return _MULTISPACE.sub(" ", cleaned.lower()).strip()


def _coerce_year(value: Any) -> int:
    if value in ("", None):
        return -1
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return -1
    return year if 0 < year < 3000 else -1


def _goodreads_series_timestamp(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    timestamp = pd.Series(-1, index=frame.index, dtype="int64")
    for column in columns:
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(
            frame[column],
            utc=True,
            errors="coerce",
            format=_GOODREADS_TS_FORMAT,
        )
        mask = parsed.notna()
        if not mask.any():
            continue
        numeric = pd.Series(-1, index=frame.index, dtype="int64")
        numeric.loc[mask] = (parsed[mask].astype("int64") // 10**9).astype("int64")
        timestamp = timestamp.where(timestamp >= 0, numeric)
    return timestamp.where(timestamp >= 0, 0).astype(int)


def _extract_goodreads_shelves(raw: Any, *, limit: int = 32) -> list[str]:
    if not isinstance(raw, list):
        return []
    cleaned = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        token = _normalize_token_block(str(name))
        if not token:
            continue
        token = token.replace(" ", "_")
        try:
            count = int(entry.get("count", 0))
        except (TypeError, ValueError):
            count = 0
        cleaned.append((count, token))
    cleaned.sort(key=lambda pair: pair[0], reverse=True)
    return [token for _, token in cleaned[:limit]]


def _flatten_series_list(values: Any) -> str:
    if isinstance(values, list):
        tokens = [_normalize_token_block(str(value)) for value in values if value]
        return " ".join(token for token in tokens if token)
    if isinstance(values, str):
        return _normalize_token_block(values)
    return ""


def _author_tokens(entries: Any, author_map: Dict[str, str]) -> str:
    if not isinstance(entries, list):
        return ""
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        author_id = entry.get("author_id")
        if not author_id:
            continue
        name = author_map.get(str(author_id))
        if name:
            names.append(name)
    if not names:
        return ""
    return _MULTISPACE.sub(" ", " ".join(names)).strip()


def _series_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    if callable(default):
        return pd.Series([default() for _ in range(len(frame))], index=frame.index)
    return pd.Series([default] * len(frame), index=frame.index)


def _year_to_timestamp(year: int | None) -> int | None:
    if year is None or year <= 0:
        return None
    dt = datetime(year, 1, 1, tzinfo=timezone.utc)
    return int(dt.timestamp())


def _http_get(url: str, *, timeout: int, stream: bool = False) -> requests.Response:
    try:
        response = requests.get(url, timeout=timeout, stream=stream)
        response.raise_for_status()
        return response
    except requests_exceptions.SSLError:
        print(
            f"[notebook_utils] SSL verification failed for {url}; retrying without verification."
        )
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
        try:
            response = requests.get(url, timeout=timeout, stream=stream, verify=False)
            response.raise_for_status()
            return response
        except requests_exceptions.RequestException:
            print("[notebook_utils] requests fallback failed; using urllib without SSL.")
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, timeout=timeout, context=context) as handle:
                data = handle.read()

            class _InlineResponse:
                def __init__(self, payload: bytes) -> None:
                    self.content = payload

                def iter_content(self, chunk_size: int):
                    for start in range(0, len(self.content), chunk_size):
                        yield self.content[start : start + chunk_size]

                def raise_for_status(self) -> None:
                    return None

            return _InlineResponse(data)  # type: ignore[return-value]


def _download_file_with_mirrors(target: Path, urls: list[str], *, required: bool = True) -> Path | None:
    if target.exists():
        return target
    last_error: Exception | None = None
    for url in urls:
        try:
            response = _http_get(url, timeout=300, stream=True)
            with target.open("wb") as fh:
                for chunk in response.iter_content(1 << 20):
                    if chunk:
                        fh.write(chunk)
            return target
        except Exception as err:  # pragma: no cover - best effort network code
            last_error = err
            if target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass
    if required:
        raise RuntimeError(f"Failed to download {target.name}: {last_error}")
    return None


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

    response = _http_get(url, timeout=120 if dataset == "medium" else 60)
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


def build_amazon_interaction_frame(
    dataset: str = "beauty",
    cache_dir: str | Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Download an Amazon reviews subset and emit the schema-compatible frame."""
    dataset = (dataset or "beauty").lower()
    base_name = _AMAZON_BASE_NAMES.get(dataset)
    if base_name is None:
        raise ValueError(f"Unknown amazon dataset '{dataset}'. Available: {AMAZON_DATASETS}")
    cache_dir = Path(cache_dir or Path(__file__).resolve().parents[2] / "data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = cache_dir / f"{base_name}_5.json.gz"
    meta_path = cache_dir / f"meta_{base_name}.json.gz"
    review_urls = [
        f"{mirror}/{quote(base_name)}_5.json.gz" for mirror in _AMAZON_REVIEW_MIRRORS
    ]
    meta_urls = [
        f"{mirror}/meta_{quote(base_name)}.json.gz" for mirror in _AMAZON_META_MIRRORS
    ]
    _download_file_with_mirrors(reviews_path, review_urls, required=True)
    _download_file_with_mirrors(meta_path, meta_urls, required=False)
    frame = pd.read_json(reviews_path, compression="gzip", lines=True)
    expected = {"reviewerID", "asin", "overall", "unixReviewTime"}
    missing = expected - set(frame.columns)
    if missing:
        raise RuntimeError(f"Amazon file {archive_path} missing required columns: {missing}")
    n_rows = len(frame)
    review_default = pd.Series([""] * n_rows)
    frame["reviewText"] = frame.get("reviewText", review_default).fillna("")
    frame["summary"] = frame.get("summary", review_default).fillna("")
    category_series = frame.get("category")
    if category_series is None:
        category_series = pd.Series([""] * n_rows)
    frame["category"] = category_series.apply(
        lambda value: " ".join(value) if isinstance(value, (list, tuple)) else str(value or "")
    )
    item_text = (
        frame["summary"].astype(str).str.strip()
        + " "
        + frame["reviewText"].astype(str).str.strip()
    ).str.strip()
    item_text = item_text.str.lower().str.replace(r"\s+", " ", regex=True)
    frame["item_text"] = item_text
    frame["item_genres"] = frame["category"].fillna("").astype(str).str.lower().str.replace(
        r"\s+", " ", regex=True
    )
    frame["item_tags"] = ""
    frame["text_len"] = frame["item_text"].str.split().apply(len).astype(int)
    frame["timestamp"] = frame["unixReviewTime"].astype(int)
    output = pd.DataFrame(
        {
            "user_id": frame["reviewerID"].astype(str),
            "item_id": frame["asin"].astype(str),
            "rating_or_y": frame["overall"].astype(float),
            "timestamp": frame["timestamp"],
            "item_text": frame["item_text"].fillna(""),
            "item_genres": frame["item_genres"],
            "item_tags": frame["item_tags"],
            "release_year": -1,
            "release_ts": -1,
            "text_len": frame["text_len"],
        }
    )
    if limit is not None and limit > 0:
        output = output.head(limit)
    return output


def build_goodreads_interaction_frame(
    genre: str = "poetry",
    cache_dir: str | Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Download a Goodreads genre subset and emit the schema-compatible frame."""
    genre = (genre or "poetry").lower()
    slug = _GOODREADS_GENRE_SLUGS.get(genre)
    if slug is None:
        raise ValueError(
            f"Unknown goodreads genre '{genre}'. Available: {GOODREADS_GENRES}"
        )
    cache_dir = Path(cache_dir or Path(__file__).resolve().parents[2] / "data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    interactions_name = f"goodreads_interactions_{slug}.json.gz"
    books_name = f"goodreads_books_{slug}.json.gz"
    authors_name = "goodreads_book_authors.json.gz"
    interactions_path = cache_dir / interactions_name
    books_path = cache_dir / books_name
    authors_path = cache_dir / authors_name
    _download_file_with_mirrors(
        interactions_path,
        [f"{GOODREADS_BASE_URL}/byGenre/{interactions_name}"],
        required=True,
    )
    _download_file_with_mirrors(
        books_path,
        [f"{GOODREADS_BASE_URL}/byGenre/{books_name}"],
        required=True,
    )
    _download_file_with_mirrors(
        authors_path,
        [f"{GOODREADS_BASE_URL}/{authors_name}"],
        required=False,
    )
    read_kwargs = {"compression": "gzip", "lines": True}
    if limit is not None and limit > 0:
        read_kwargs["nrows"] = int(limit)
    interactions = pd.read_json(interactions_path, **read_kwargs)
    if limit is not None and limit > 0:
        interactions = interactions.head(limit)
    if interactions.empty:
        raise RuntimeError("No interactions found in the Goodreads file.")
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["book_id"] = interactions["book_id"].astype(str)
    timestamp_cols = ["date_updated", "read_at", "date_added", "started_at"]
    interactions["timestamp"] = _goodreads_series_timestamp(interactions, timestamp_cols)
    rating_col = interactions.get("rating")
    if rating_col is None:
        interactions["rating"] = 0.0
    interactions["rating"] = interactions["rating"].fillna(0).astype(float)
    books = pd.read_json(books_path, compression="gzip", lines=True)
    books["book_id"] = books["book_id"].astype(str)
    book_ids = interactions["book_id"].unique().tolist()
    books = books[books["book_id"].isin(book_ids)].copy()
    title_series = _series_or_default(books, "title_without_series", "")
    title_fallback = _series_or_default(books, "title", "")
    books["title_norm"] = (
        title_series.fillna(title_fallback).fillna("").apply(_normalize_free_text)
    )
    books["description_norm"] = _series_or_default(books, "description", "").apply(
        _normalize_free_text
    )
    books["publisher_norm"] = _series_or_default(books, "publisher", "").apply(
        _normalize_token_block
    )
    books["format_norm"] = _series_or_default(books, "format", "").apply(
        _normalize_token_block
    )
    books["series_norm"] = _series_or_default(books, "series", list).apply(
        _flatten_series_list
    )
    books["shelf_tokens"] = _series_or_default(books, "popular_shelves", list).apply(
        _extract_goodreads_shelves
    )
    author_map: Dict[str, str] = {}
    if authors_path.exists():
        author_ids: set[str] = set()
        for entries in _series_or_default(books, "authors", list):
            if not isinstance(entries, list):
                continue
            for entry in entries:
                author_id = entry.get("author_id")
                if author_id:
                    author_ids.add(str(author_id))
        if author_ids:
            authors_df = pd.read_json(authors_path, compression="gzip", lines=True)
            authors_df["author_id"] = authors_df["author_id"].astype(str)
            authors_df["name"] = authors_df["name"].astype(str).str.lower()
            filtered = authors_df[authors_df["author_id"].isin(author_ids)]
            author_map = filtered.set_index("author_id")["name"].to_dict()
    books["author_text"] = _series_or_default(books, "authors", list).apply(
        lambda value: _author_tokens(value, author_map)
    )
    books["shelf_phrase"] = books["shelf_tokens"].apply(lambda tokens: " ".join(tokens))
    books["item_genres"] = books["shelf_tokens"].apply(
        lambda tokens: " ".join(tokens[:12])
    )
    books["item_tags"] = books["shelf_tokens"].apply(
        lambda tokens: " ".join(tokens[12:32])
    )
    text_columns = [
        "title_norm",
        "author_text",
        "description_norm",
        "series_norm",
        "publisher_norm",
        "format_norm",
        "shelf_phrase",
    ]
    books["item_text"] = (
        books[text_columns]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    books["item_text"] = books["item_text"].fillna("")
    books["text_len"] = books["item_text"].str.split().apply(len).astype(int)
    publication_years = _series_or_default(books, "publication_year", -1)
    books["release_year"] = publication_years.apply(_coerce_year)
    books["release_ts"] = books["release_year"].apply(
        lambda year: _year_to_timestamp(year if year > 0 else None) or -1
    )
    books["release_ts"] = books["release_ts"].astype(int)
    frame = interactions.merge(
        books[
            [
                "book_id",
                "item_text",
                "item_genres",
                "item_tags",
                "release_year",
                "release_ts",
                "text_len",
            ]
        ],
        on="book_id",
        how="left",
    )
    frame = frame.rename(
        columns={
            "book_id": "item_id",
            "rating": "rating_or_y",
        }
    )
    frame["item_text"] = frame["item_text"].fillna("")
    frame["item_genres"] = frame["item_genres"].fillna("")
    frame["item_tags"] = frame["item_tags"].fillna("")
    frame["release_year"] = frame["release_year"].fillna(-1).astype(int)
    frame["release_ts"] = frame["release_ts"].fillna(-1).astype(int)
    frame["text_len"] = frame["text_len"].fillna(0).astype(int)
    frame["timestamp"] = frame["timestamp"].astype(int)
    frame["rating_or_y"] = frame["rating_or_y"].astype(float)
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


def _ensure_msnews_resources(variant: str, cache_dir: Path) -> tuple[Path, Path]:
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    info = _MSNEWS_VARIANTS[variant]
    zip_path = cache_dir / info["zip_name"]
    if not zip_path.exists():
        try:
            _download_file_with_mirrors(zip_path, [info["url"]])
        except Exception as err:
            raise RuntimeError(
                "Microsoft News (MIND) downloads require accepting Microsoft's license. "
                f"Download {info['zip_name']} manually from https://msnews.github.io/ "
                f"and place it under {zip_path.parent} before re-running the notebook."
            ) from err
    extract_root = cache_dir / info["folder"]
    news_path = extract_root / "news.tsv"
    behaviors_path = extract_root / "behaviors.tsv"
    if not news_path.exists() or not behaviors_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
    if not news_path.exists() or not behaviors_path.exists():
        raise FileNotFoundError(
            f"Missing Microsoft News files for {variant}; expected {news_path} and {behaviors_path}"
        )
    return news_path, behaviors_path


def _extract_msnews_clicked(impressions: str) -> list[str]:
    results: list[str] = []
    if not isinstance(impressions, str):
        return results
    for token in impressions.split():
        if "-" not in token:
            continue
        item_id, flag = token.rsplit("-", 1)
        if flag == "1" and item_id:
            results.append(item_id)
    return results


def build_msnews_interaction_frame(
    variant: str = "mind_small_train",
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir or Path(__file__).resolve().parents[2] / "data")
    normalized_variant = (variant or "mind_small_train").lower()
    if normalized_variant not in _MSNEWS_VARIANTS:
        raise ValueError(
            f"Unknown Microsoft News variant '{variant}'. "
            f"Available options: {', '.join(MSNEWS_VARIANTS)}"
        )
    news_path, behaviors_path = _ensure_msnews_resources(
        normalized_variant, cache_dir
    )
    news_columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    news_df = pd.read_csv(
        news_path,
        sep="\t",
        header=None,
        names=news_columns,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        encoding="utf-8",
        engine="python",
    )
    news_df["news_id"] = news_df["news_id"].astype(str)
    title_norm = news_df["title"].fillna("").apply(_normalize_free_text)
    abstract_norm = news_df["abstract"].fillna("").apply(_normalize_free_text)
    news_df["item_text"] = (title_norm + " " + abstract_norm).str.strip()
    news_df["item_genres"] = news_df["category"].fillna("").apply(_normalize_token_block)
    news_df["item_tags"] = news_df["subcategory"].fillna("").apply(
        _normalize_token_block
    )
    news_df["text_len"] = news_df["item_text"].str.split().apply(len).astype(int)
    news_df = news_df.rename(columns={"news_id": "item_id"})

    behavior_columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "history",
        "impressions",
    ]
    behaviors = pd.read_csv(
        behaviors_path,
        sep="\t",
        header=None,
        names=behavior_columns,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        encoding="utf-8",
        engine="python",
    )
    behaviors["timestamp"] = pd.to_datetime(
        behaviors["timestamp"],
        format=_MSNEWS_TIMESTAMP_FORMAT,
        errors="coerce",
    )
    behaviors = behaviors.dropna(subset=["timestamp"])
    behaviors["timestamp"] = (
        behaviors["timestamp"].astype("int64") // 1_000_000_000
    ).astype("int64")
    behaviors["user_id"] = behaviors["user_id"].astype(str)

    interaction_rows: list[tuple[str, str, int]] = []
    for row in behaviors.itertuples(index=False):
        clicked = _extract_msnews_clicked(row.impressions)
        if not clicked:
            continue
        for item_id in clicked:
            interaction_rows.append((row.user_id, item_id, int(row.timestamp)))
    if not interaction_rows:
        raise RuntimeError(
            f"No clicks parsed from Microsoft News behaviors at {behaviors_path}"
        )
    interactions = pd.DataFrame(
        interaction_rows, columns=["user_id", "item_id", "timestamp"]
    )
    interactions["rating_or_y"] = 1.0

    merged = interactions.merge(
        news_df[["item_id", "item_text", "item_genres", "item_tags", "text_len"]],
        on="item_id",
        how="left",
    )
    merged["item_text"] = merged["item_text"].fillna("")
    merged["item_genres"] = merged["item_genres"].fillna("")
    merged["item_tags"] = merged["item_tags"].fillna("")
    merged["text_len"] = merged["text_len"].fillna(0).astype(int)
    merged["release_year"] = -1
    merged["release_ts"] = -1
    merged["timestamp"] = merged["timestamp"].astype(int)
    merged["rating_or_y"] = merged["rating_or_y"].astype(float)
    return merged[
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
    "build_amazon_interaction_frame",
    "build_goodreads_interaction_frame",
    "build_msnews_interaction_frame",
    "MOVIELENS_SMALL_URL",
    "MOVIELENS_MEDIUM_URL",
    "AMAZON_DATASETS",
    "GOODREADS_GENRES",
    "MSNEWS_VARIANTS",
]
