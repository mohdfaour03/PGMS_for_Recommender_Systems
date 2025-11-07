"""Utilities for reading and writing interaction data."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any, Sequence

REQUIRED_COLUMNS = ["user_id", "item_id", "rating_or_y", "item_text"]
INT_COLUMNS = {"timestamp", "release_year", "release_ts", "text_len"}
FLOAT_COLUMNS: set[str] = set()
STRING_COLUMNS = {"item_genres", "item_tags"}


class DataFormatError(ValueError):
    """Raised when the input data does not match the expected schema."""


def _ensure_required_columns(row: Dict[str, Any]) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in row]
    if missing:
        raise DataFormatError(f"Missing required columns: {missing}")


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        intval = int(value)
        return intval if intval >= 0 else None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    intval = int(float(text))
    return intval if intval >= 0 else None


def _maybe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _writer_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not rows:
        return list(REQUIRED_COLUMNS)
    ordered: List[str] = []
    seen = set()
    for column in REQUIRED_COLUMNS:
        if any(column in row for row in rows):
            ordered.append(column)
            seen.add(column)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered


def load_interactions(path: str | Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """Load interactions from a CSV file.

    Parameters
    ----------
    path:
        Location of the input file. CSV is supported out of the box; attempting
        to read other formats raises a :class:`NotImplementedError`.
    """
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise NotImplementedError(
            f"Unsupported extension '{path.suffix}'. Only CSV is supported in the"
            " reference implementation."
        )

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")

    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            _ensure_required_columns(row)
            parsed: Dict[str, Any] = {
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "rating_or_y": float(row["rating_or_y"]),
                "item_text": row["item_text"],
            }
            for column in reader.fieldnames or []:
                if column in parsed:
                    continue
                value = row.get(column)
                if column in INT_COLUMNS:
                    parsed[column] = _maybe_int(value)
                elif column in FLOAT_COLUMNS:
                    parsed[column] = float(value) if value not in {None, ""} else None
                elif column in STRING_COLUMNS:
                    parsed[column] = _maybe_str(value)
                else:
                    if value not in {None, ""}:
                        parsed[column] = value
            rows.append(parsed)
            if limit is not None and len(rows) >= limit:
                break
    print(f"Loaded {len(rows)} interactions from {path}.")
    if rows:
        n_users = len({row["user_id"] for row in rows})
        n_items = len({row["item_id"] for row in rows})
        print(
            f"Dataset contains {n_users} unique users and {n_items} unique items.")
    return rows


def save_interactions_csv(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    fieldnames = _writer_fieldnames(materialized)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            serialized: Dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                serialized[key] = "" if value is None else value
            writer.writerow(serialized)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_matrix(matrix: list[list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(matrix, fh)


def load_matrix(path: str | Path) -> list[list[float]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [[float(value) for value in row] for row in data]


def save_text_lines(lines: Iterable[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(f"{line}\n")


def read_text_lines(path: str | Path) -> List[str]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]
