# Cold-Start Recommender Sandbox

This package contains a minimal, dependency-light implementation of a strict cold-start recommender workflow. It demonstrates how to split interactions, vectorise item text without leakage, and evaluate lightweight content-to-factor models such as CTR-lite (linear regression), A2F (a shallow MLP), CTPF (topic model coupling), CDL (denoising autoencoder), and HFT (topic-linked latent factors) that map item text into collaborative filtering latent factors.

## Data schema

The expected interaction file (CSV) must contain the following columns:

| column        | description                                      |
| ------------- | ------------------------------------------------ |
| `user_id`     | Unique identifier for the user.                  |
| `item_id`     | Unique identifier for the item.                  |
| `rating_or_y` | Explicit rating or implicit target signal.      |
| `item_text`   | Natural-language description of the item.       |

For a practical benchmark you can fetch MovieLens latest-small or latest (≈1M interactions) via the notebook helpers in `coldstart/src/notebook_utils.py`. The default notebook workflow now downloads the medium release (`ml-latest`, ≈1M interactions) and materialises it as `coldstart/data/movielens_latest_medium.csv`. Switch the `dataset` parameter in the notebook if you want the smaller quick-test variant.

## Quickstart

1. **Open the notebook**

   Launch `coldstart/notebooks/coldstart_workflow.ipynb`. The first cell ensures the MovieLens latest (medium) dataset is present.

2. **Prepare the data**

   The notebook calls `pipeline.prepare_dataset` with TF-IDF parameters from `configs/base.yaml`, generating warm/cold splits plus text features under `coldstart/output/notebook_run_*`. By default we cap preparation to the first 1.2 M interactions via the `prepare.interaction_limit` setting to keep RAM usage manageable—tweak this value if you need more data.

3. **Train and evaluate**

   Set `model_choice` to any of `ctrlite`, `a2f`, `ctpf`, `cdl`, `hft`, `micm`, or `all` and execute the training cell. Results (Hit@K, NDCG@K) are returned as a JSON-like dictionary, and model artefacts are written alongside the prepared data. The `micm` option introduces an experimental mutual-information contrastive mapper that aligns text features with collaborative factors via an InfoNCE loss.

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
)
print(metrics)
```

All configurables (MF hyperparameters, TF-IDF settings, content-model knobs) live in `configs/base.yaml`.

## Repository workflow

This project is delivered as source code only; the automation stops at committing changes to the local Git repository in the execution environment. Publishing to an external remote (for example, pushing to GitHub) is intentionally left to the user so credentials never need to be embedded in the tooling.

