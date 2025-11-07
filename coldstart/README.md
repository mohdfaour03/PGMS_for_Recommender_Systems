# Cold-Start Recommender Sandbox

This package contains a minimal, dependency-light implementation of a strict cold-start recommender workflow. It demonstrates how to split interactions, vectorise item text without leakage, and evaluate lightweight content-to-factor models such as CTR-lite (linear regression), A2F (a shallow MLP), CTPF (topic model coupling), CDL (denoising autoencoder), and HFT (topic-linked latent factors) that map item text into collaborative filtering latent factors.

## Data schema

The expected interaction file (CSV) must contain the following columns:

| column          | description                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| `user_id`       | Unique identifier for the user.                                             |
| `item_id`       | Unique identifier for the item.                                             |
| `rating_or_y`   | Explicit rating or implicit target signal.                                 |
| `timestamp`     | Unix timestamp of the interaction (drives recency-aware negative sampling). |
| `item_text`     | Leakage-safe text (title + genres + tags, lowercased, release year removed).|
| `item_genres`   | Space-delimited genre tokens.                                               |
| `item_tags`     | Space-delimited tag tokens aggregated per item (if available).              |
| `release_year`  | Four-digit release year parsed from the title (`-1` when unknown).          |
| `release_ts`    | Unix timestamp anchoring the release year (Jan 1).                          |
| `text_len`      | Token count of the normalised `item_text`.                                  |

For a practical benchmark you can fetch MovieLens latest-small or latest (~1 M interactions) via the notebook helpers in `coldstart/src/notebook_utils.py`. The default notebook workflow now downloads the medium release (`ml-latest`, ~1 M interactions) and materialises it as `coldstart/data/movielens_latest_medium.csv`. Switch the `dataset` parameter in the notebook if you want the smaller quick-test variant. The loader now attaches timestamps, genres, tags, release-year anchors, and normalised text so downstream exposure modelling and CMCL have everything they need.

## Quickstart

1. **Open the notebook**

   Launch `coldstart/notebooks/coldstart_workflow.ipynb`. The first cell ensures the MovieLens latest (medium) dataset is present.

2. **Prepare the data**

   The notebook calls `pipeline.prepare_dataset` with TF-IDF parameters from `configs/base.yaml`, generating warm/cold splits plus text features under `coldstart/output/notebook_run_*`. By default we cap preparation to the first 1.2 M interactions via the `prepare.interaction_limit` setting to keep RAM usage manageable—tweak this value if you need more data.

3. **Train and evaluate**

   Set `model_choice` to any of `ctrlite`, `a2f`, `ctpf`, `cdl`, `hft`, `micm`, `cmcl`, or `all` and execute the training cell. Results now include point estimates, bootstrap 95 % confidence intervals, per-bucket metrics (item text length, item popularity, user history length), and—when MICM is also in the run—`delta_vs_micm` summaries. `micm` is the mutual-information contrastive mapper introduced previously. `cmcl` adds counterfactual multi-positive contrastive learning with inverse-propensity reweighting, optional self-normalisation, top-k focal reweighting, and semantic hard negatives mined from text features. You can evaluate multiple cutoffs at once by passing a list to `k_eval` (for example, [10, 20, 50]).

The same functions can be scripted directly in Python if you prefer not to use the notebook:

```python
from coldstart.src import pipeline
from coldstart.src.notebook_utils import build_interaction_frame, _read_simple_yaml

# Prepare dataset
df = build_interaction_frame(dataset="medium")
df.to_csv("coldstart/data/movielens_latest_medium.csv", index=False)
config = _read_simple_yaml("coldstart/configs/base.yaml")
pipeline.prepare_dataset(
    "coldstart/data/movielens_latest_medium.csv",
    "coldstart/output/example_run",
    tfidf_params=config.get("tfidf", {}),
    cold_item_frac=0.15,
    seed=42,
)

# Train and evaluate all models
metrics = pipeline.train_and_evaluate_content_model(
    "coldstart/output/example_run",
    model="all",
    a2f_cfg=config.get("a2f", {}),
    ctpf_cfg=config.get("ctpf", {}),
    cdl_cfg=config.get("cdl", {}),
    hft_cfg=config.get("hft", {}),
    micm_cfg=config.get("micm", {}),
    cmcl_cfg=config.get("cmcl", {}),
)
print(metrics)
```

All configurables (MF hyperparameters, TF-IDF settings, content-model knobs) live in `configs/base.yaml`.


## CMCL + pseudo-exposure modeling

The CMCL block in `configs/base.yaml` describes both the downstream contrastive learner and the upstream exposure estimator:

- **Exposure estimator** (`cmcl.exposure`) learns a propensity model by sampling timestamp-aligned negatives per interaction, extracting popularity/recency/text-length/genre features, and fitting a logistic/MLP with optional GPU acceleration. Checkpoints are saved under `output_run/models/exposure.ckpt` and reused unless you delete them or override the `path`.
- **CMCL trainer** consumes MF user factors, the warm text encoder, inverse-propensity weights, and optional semantic hard negatives mined via cosine similarity on text features. It supports IPS self-normalisation, top-k focal weighting, and a cap on the number of positives sampled per user.

Invoke `cmcl` alongside MICM to automatically report deltas plus per-bucket breakdowns that highlight where counterfactual training helps the most (for example, low-popularity or short-text items). If the exposure checkpoint is missing, it is trained on-the-fly; otherwise it is reused for faster sweeps.

## Evaluation output

`pipeline.train_and_evaluate_content_model` now returns richer dictionaries:

- `hit@K`, `ndcg@K`, and their `_ci` counterparts (bootstrap 95 % confidence intervals).
- `buckets`: nested dictionaries keyed by `item_text_len`, `item_popularity`, and `user_history_len` with per-bucket metrics.
- `delta_vs_micm`: automatically populated for every model when MICM is part of the run, including bucket-level deltas to facilitate the “apples-to-apples” comparison highlighted in the acceptance checklist.


## Repository workflow

This project is delivered as source code only; the automation stops at committing changes to the local Git repository in the execution environment. Publishing to an external remote (for example, pushing to GitHub) is intentionally left to the user so credentials never need to be embedded in the tooling.

