from pathlib import Path
import json
from coldstart.src import pipeline
from coldstart.src.notebook_utils import build_goodreads_interaction_frame, _read_simple_yaml

# Define paths
DATA_DIR = Path("coldstart/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
GOODREADS_PATH = DATA_DIR / "goodreads_poetry.csv"
RUN_DIR = Path("coldstart/output/goodreads_run")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# 1. Download/Load Data
if not GOODREADS_PATH.exists():
    print("Downloading/Loading Goodreads Poetry dataset...")
    try:
        # Limit to 300k to match the "easy/fast" requirement
        df = build_goodreads_interaction_frame(genre="poetry", limit=300000)
        df.to_csv(GOODREADS_PATH, index=False)
        print(f"Saved Goodreads Poetry dataset to {GOODREADS_PATH}")
    except Exception as e:
        print(f"Error loading Goodreads Poetry dataset: {e}")
        exit(1)
else:
    print("Goodreads Poetry dataset already exists.")

# 2. Load Config
config = _read_simple_yaml("coldstart/configs/base.yaml")

# 3. Prepare Dataset
print("Preparing dataset...")
pipeline.prepare_dataset(
    GOODREADS_PATH,
    RUN_DIR,
    tfidf_params=config.get("tfidf", {}),
    cold_item_frac=0.2,
    seed=42,
    interaction_limit=50000, 
)

# 4. Train and Evaluate
print("Training and evaluating models...")
# Using reduced iterations for speed as requested ("dont train it for tooooo long")
metrics = pipeline.train_and_evaluate_content_model(
    RUN_DIR,
    k_factors=16,
    k_eval=5,
    mf_reg=float(config.get("mf", {}).get("reg", 0.02)),
    mf_iters=10, 
    mf_lr=float(config.get("mf", {}).get("lr", 0.02)),
    seed=42,
    ctrlite_reg=float(config.get("ctrlite", {}).get("reg", 0.01)),
    ctrlite_lr=float(config.get("ctrlite", {}).get("lr", 0.1)),
    ctrlite_iters=20, 
    adaptive=False,
    model="ctrlite,cdl,cmcl", 
    a2f_cfg=config.get("a2f", {}),
    ctpf_cfg=config.get("ctpf", {}),
    cdl_cfg={"iters": 50}, 
    hft_cfg=config.get("hft", {}),
    micm_cfg=config.get("micm", {}),
    cmcl_cfg={"iters": 3}, 
    backend="torch",
    prefer_gpu=False, 
    mf_cfg={
        "batch_size": 512, 
        "score_batch_size": 512,
        "infer_batch_size": 512,
        "ctrlite_batch_size": 512,
    },
)

print("\n=== Final Results ===")
print(json.dumps(metrics, indent=2))
with open(RUN_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
