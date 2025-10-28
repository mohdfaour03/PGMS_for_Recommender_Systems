"""Strict cold-item splits for recommender experiments."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

from . import data_io


def strict_cold_split(
    interactions: Sequence[dict],
    cold_item_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[str]]:
    """Split interactions into warm and cold subsets.

    The function keeps a fixed portion of items for the cold test set and
    returns warm interactions, cold interactions, and the list of cold item
    identifiers.
    """
    if not 0.0 < cold_item_frac < 1.0:
        raise ValueError("cold_item_frac must be in (0, 1)")

    items = sorted({row["item_id"] for row in interactions})
    if not items:
        return [], [], []
    rng = random.Random(seed)
    n_cold = max(1, int(round(len(items) * cold_item_frac)))
    cold_items = set(rng.sample(items, n_cold))

    warm_rows: list[dict] = []
    cold_rows: list[dict] = []
    for row in interactions:
        if row["item_id"] in cold_items:
            cold_rows.append(row)
        else:
            warm_rows.append(row)

    assert not any(row["item_id"] in cold_items for row in warm_rows), (
        "Cold item leakage detected in warm interactions!"
    )
    return warm_rows, cold_rows, sorted(cold_items)


def persist_split(
    warm_rows: Iterable[dict],
    cold_rows: Iterable[dict],
    cold_items: Iterable[str],
    out_dir: str | Path,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data_io.save_interactions_csv(warm_rows, out_path / "warm_interactions.csv")
    data_io.save_interactions_csv(cold_rows, out_path / "cold_interactions.csv")
    data_io.save_text_lines(cold_items, out_path / "cold_item_ids.txt")
