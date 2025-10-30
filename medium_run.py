from pathlib import Path
from coldstart.src import pipeline
from coldstart.src.notebook_utils import build_interaction_frame, _read_simple_yaml

MEDIUM_PATH = Path("coldstart/data/movielens_latest_medium.csv")
if not MEDIUM_PATH.exists():
    print("Downloading MovieLens medium dataset...")
    MEDIUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build_interaction_frame(dataset="medium")
    df.to_csv(MEDIUM_PATH, index=False)
    print(f"Saved medium dataset to {MEDIUM_PATH}")
else:
    print("Medium dataset already downloaded.")

config = _read_simple_yaml("coldstart/configs/base.yaml")
run_dir = Path("coldstart/output/medium_run")
run_dir.mkdir(parents=True, exist_ok=True)

pipeline.prepare_dataset(
    MEDIUM_PATH,
    run_dir,
    tfidf_params=config.get("tfidf", {}),
    cold_item_frac=0.2,
    seed=42,
    interaction_limit=300000,
)

metrics = pipeline.train_and_evaluate_content_model(
    run_dir,
    k_factors=16,
    k_eval=5,
    mf_reg=float(config.get("mf", {}).get("reg", 0.02)),
    mf_iters=int(config.get("mf", {}).get("iters", 30)),
    mf_lr=float(config.get("mf", {}).get("lr", 0.02)),
    seed=42,
    ctrlite_reg=float(config.get("ctrlite", {}).get("reg", 0.01)),
    ctrlite_lr=float(config.get("ctrlite", {}).get("lr", 0.1)),
    ctrlite_iters=int(config.get("ctrlite", {}).get("iters", 80)),
    adaptive=False,
    model="all",
    a2f_cfg=config.get("a2f", {}),
    ctpf_cfg=config.get("ctpf", {}),
    cdl_cfg=config.get("cdl", {}),
    hft_cfg=config.get("hft", {}),
    backend="torch",
    prefer_gpu=True,
    mf_cfg={
        "batch_size": 8192,
        "score_batch_size": 8192,
        "infer_batch_size": 8192,
        "ctrlite_batch_size": 4096,
    },
)

import json
print(json.dumps(metrics, indent=2))
