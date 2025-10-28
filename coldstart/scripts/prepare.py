#!/usr/bin/env python3
"""Data preparation entrypoint for the cold-start benchmark."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coldstart.src import pipeline


def _read_simple_yaml(path: str | Path) -> dict:
    content = Path(path).read_text(encoding="utf-8").splitlines()
    config: dict = {}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True, help="Path to the raw interactions CSV")
    parser.add_argument("--out_dir", required=True, help="Directory to place prepared assets")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"))
    parser.add_argument("--cold_item_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = _read_simple_yaml(args.config)
    tfidf_params = config.get("tfidf", {})
    pipeline.prepare_dataset(
        args.data_path,
        args.out_dir,
        tfidf_params=tfidf_params,
        cold_item_frac=args.cold_item_frac,
        seed=args.seed,
    )
    print("Preparation complete.")


if __name__ == "__main__":
    main()
