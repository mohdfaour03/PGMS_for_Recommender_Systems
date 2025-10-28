#!/usr/bin/env python3
"""Train models and evaluate on the cold split."""
from __future__ import annotations

import argparse
import ast
import json
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
    parser.add_argument("--data_dir", required=True, help="Prepared data directory")
    parser.add_argument("--model", default="ctrlite", choices=["ctrlite", "fm"])
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"))
    parser.add_argument("--k", type=int, default=32, help="Latent dimensionality")
    parser.add_argument("--K", type=int, default=10, help="Evaluation cutoff")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--adaptive", action="store_true", help="Run adaptive variant as well")
    args = parser.parse_args()

    config = _read_simple_yaml(args.config)

    if args.model == "fm":
        print(json.dumps({"model": "fm", "status": "skipped", "reason": "Factorization machine backend not available"}))
        return

    tfidf_cfg = config.get("tfidf", {})
    mf_cfg = config.get("mf", {})
    ctrlite_cfg = config.get("ctrlite", {})

    results = pipeline.train_and_evaluate_ctrlite(
        args.data_dir,
        k_factors=args.k,
        k_eval=args.K,
        mf_reg=float(mf_cfg.get("reg", 0.1)),
        mf_iters=int(mf_cfg.get("iters", 5)),
        mf_lr=float(mf_cfg.get("lr", 0.01)),
        seed=args.split_seed,
        ctrlite_reg=float(ctrlite_cfg.get("reg", 1.0)),
        ctrlite_lr=float(ctrlite_cfg.get("lr", 0.1)),
        ctrlite_iters=int(ctrlite_cfg.get("iters", 50)),
        adaptive=args.adaptive,
    )

    payload = {
        "model": args.model,
        "metrics": results,
        "config": {
            "k": args.k,
            "K": args.K,
            "split_seed": args.split_seed,
            "tfidf": tfidf_cfg,
            "mf": mf_cfg,
            "ctrlite": ctrlite_cfg,
            "adaptive": args.adaptive,
        },
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
