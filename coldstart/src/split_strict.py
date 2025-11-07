"""Strict cold-item splits for recommender experiments."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

from . import data_io


def strict_cold_split(
    interactions: Sequence[dict],
    cold_item_frac: float = 0.15,
    val_item_frac: float | None = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], list[str], list[str]]:
    """Split interactions into warm and cold subsets.

    The function keeps a fixed portion of items for the cold test set and
    returns warm interactions, cold interactions, and the list of cold item
    identifiers.
    """
    if not 0.0 < cold_item_frac < 1.0:
        raise ValueError("cold_item_frac must be in (0, 1)")
    if val_item_frac is not None and not 0.0 < val_item_frac < 1.0:
        raise ValueError("val_item_frac must be in (0, 1)")

    items = sorted({row["item_id"] for row in interactions})
    if not items:
        return [], [], [], [], []
    rng = random.Random(seed)
    n_cold = max(1, int(round(len(items) * cold_item_frac)))
    cold_items = set(rng.sample(items, n_cold))
    remaining = [item for item in items if item not in cold_items]

    val_items: set[str] = set()
    if val_item_frac:
        val_rng = random.Random(seed + 1)
        n_val = max(1, int(round(len(items) * val_item_frac)))
        n_val = min(n_val, len(remaining))
        if n_val > 0:
            val_items = set(val_rng.sample(remaining, n_val))

    warm_rows: list[dict] = []
    val_rows: list[dict] = []
    cold_rows: list[dict] = []
    for row in interactions:
        if row["item_id"] in cold_items:
            cold_rows.append(row)
        elif row["item_id"] in val_items:
            val_rows.append(row)
        else:
            warm_rows.append(row)

    warm_users = {row["user_id"] for row in warm_rows}
    if not warm_users:
        raise RuntimeError("All users became cold-only; adjust cold/val fractions.")

    def _filter_rows(rows: list[dict]) -> list[dict]:
        return [row for row in rows if row["user_id"] in warm_users]

    val_rows = _filter_rows(val_rows)
    cold_rows = _filter_rows(cold_rows)

    assert not any(row["item_id"] in cold_items for row in warm_rows), (
        "Cold item leakage detected in warm interactions!"
    )
    return warm_rows, val_rows, cold_rows, sorted(val_items), sorted(cold_items)


def persist_split(
    warm_rows: Iterable[dict],
    cold_rows: Iterable[dict],
    cold_items: Iterable[str],
    out_dir: str | Path,
    val_rows: Iterable[dict] | None = None,
    val_items: Iterable[str] | None = None,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data_io.save_interactions_csv(warm_rows, out_path / "warm_interactions.csv")
    if val_rows is not None:
        data_io.save_interactions_csv(val_rows, out_path / "val_interactions.csv")
    data_io.save_interactions_csv(cold_rows, out_path / "cold_interactions.csv")
    if val_items is not None:
        data_io.save_text_lines(val_items, out_path / "val_item_ids.txt")
    data_io.save_text_lines(cold_items, out_path / "cold_item_ids.txt")
