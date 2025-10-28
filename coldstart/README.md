# Cold-Start Recommender Sandbox

This package contains a minimal, dependency-light implementation of a strict
cold-start recommender workflow. It demonstrates how to split interactions,
vectorise item text without leakage, and evaluate a simple CTR-lite model that
maps text features into collaborative filtering latent factors.

## Data schema

The expected interaction file (CSV) must contain the following columns:

| column        | description                                      |
| ------------- | ------------------------------------------------ |
| `user_id`     | Unique identifier for the user.                  |
| `item_id`     | Unique identifier for the item.                  |
| `rating_or_y` | Explicit rating or implicit target signal.      |
| `item_text`   | Natural-language description of the item.       |

A tiny sample dataset is available in `data/sample_interactions.csv`.

## Quickstart

1. **Prepare the data**

   ```bash
   python coldstart/scripts/prepare.py \
       --data_path coldstart/data/sample_interactions.csv \
       --out_dir /tmp/coldstart_prep \
       --cold_item_frac 0.15 \
       --seed 42
   ```

2. **Train and evaluate CTR-lite**

   ```bash
   python coldstart/scripts/train_eval.py \
       --data_dir /tmp/coldstart_prep \
       --model ctrlite \
       --k 16 \
       --K 10 \
       --split_seed 42
   ```

   The command creates the following artefacts inside the output directory:

   - `warm_interactions.csv` and `cold_interactions.csv` — strict split with
     zero cold-item leakage into the warm portion.
   - `cold_item_ids.txt` — newline-delimited identifiers for the cold items.
   - `warm_item_text_features.json` / `cold_item_text_features.json` — TF-IDF
     matrices fitted on warm item text only.
   - `tfidf_state.json`, `warm_item_ids.json`, `cold_item_ids.json` — metadata
     required to reuse the prepared assets.

Both commands rely on the lightweight configuration stored in
`configs/base.yaml`. The `train_eval.py` script prints a single JSON line with
metrics such as Hit@10 and NDCG@10 on the cold-start items and writes the MF
factors to `<data_dir>/models`. Passing `--adaptive` produces an additional
adaptive variant based on rarity-aware blending of warm item factors.

## Repository workflow

This project is delivered as source code only; the automation stops at committing
changes to the local Git repository in the execution environment. Publishing to
an external remote (for example, pushing to GitHub) is intentionally left to the
user so credentials never need to be embedded in the tooling.
