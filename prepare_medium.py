from pathlib import Path
from coldstart.src import pipeline
from coldstart.src.notebook_utils import _read_simple_yaml, build_interaction_frame

DATA_PATH = Path("coldstart/data/movielens_latest_medium.csv")
RUN_DIR = Path("coldstart/output/medium_run_seed42")
RUN_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    print("Downloading MovieLens medium interactions...")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = build_interaction_frame(dataset="medium")
    frame.to_csv(DATA_PATH, index=False)
    print(f"Saved dataset to {DATA_PATH}")

config = _read_simple_yaml("coldstart/configs/base.yaml")
tfidf_params = config.get("tfidf", {})

pipeline.prepare_dataset(
    DATA_PATH,
    RUN_DIR,
    tfidf_params=tfidf_params,
    cold_item_frac=0.2,
    seed=42,
    interaction_limit=300000,
)
