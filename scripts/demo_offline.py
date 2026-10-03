"""End-to-end offline demo: synthetic data -> train -> predict a fake repo."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
from scripts.make_demo_data import *  # noqa - writes data/repos.csv
from scripts.train import main as train_main

# Train on synthetic data
train_main("data/repos.csv")

# --- Simulate a prediction on a new repo WITHOUT hitting the GitHub API ---
fake_readme = "Send 1 ETH get 2 back - guaranteed profit airdrop! Double your money now, no risk."
fake_meta = {"full_name": "eth2x/double-money", "description": "guaranteed profit airdrop",
             "stars": 2, "forks": 0, "watchers": 1, "open_issues": 3, "size_kb": 15,
             "age_days": 1, "has_pages": 1, "fork": 0, "archived": 0,
             "num_topics": 0, "license_present": 0}

from config.config import MODEL_DIR
from src.text_features import TextScamModel
from src.meta_features import MetaScamModel
from src.fusion import fuse, verdict

text_p = TextScamModel.load(MODEL_DIR / "text_model.joblib").predict_proba([fake_readme])[0]
meta_p = MetaScamModel.load(MODEL_DIR / "meta_model.joblib").predict_proba([fake_meta])[0]
weights = json.load(open(MODEL_DIR / "fusion.json"))["weights"]
fused = float(fuse([text_p], [meta_p], [0.0], weights)[0])

print("\n=== DEMO PREDICTION ===")
print(json.dumps({"repo": "eth2x/double-money",
                  "text_score": round(text_p, 3), "meta_score": round(meta_p, 3),
                  "fused_score": round(fused, 3), "verdict": verdict(fused)}, indent=2))
