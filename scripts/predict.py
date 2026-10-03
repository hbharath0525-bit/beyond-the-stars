"""Step 3 - Predict whether a GitHub repo is a scam (multimodal)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np

from config.config import MODEL_DIR
from src.github_client import GitHubRepoMiner
from src.text_features import TextScamModel
from src.meta_features import MetaScamModel
from src.image_features import ImageScamModel
from src.fusion import fuse, verdict, SCAM_THRESHOLD


def predict_repo(url: str, token: str | None = None):
    miner = GitHubRepoMiner(token)
    mined = miner.mine(url)

    text_p = TextScamModel.load(MODEL_DIR / "text_model.joblib").predict_proba([mined["readme"]])[0]
    meta_p = MetaScamModel.load(MODEL_DIR / "meta_model.joblib").predict_proba([mined["metadata"]])[0]
    imgs = mined["images"]
    image_p = float(ImageScamModel.load(MODEL_DIR / "image_model.joblib").predict_proba(imgs).max(initial=0.0)) if imgs else 0.0

    weights = json.load(open(MODEL_DIR / "fusion.json"))["weights"]
    fused = float(fuse([text_p], [meta_p], [image_p], weights)[0])

    report = {
        "repo": mined["metadata"]["full_name"],
        "url": url,
        "text_score": round(text_p, 3),
        "meta_score": round(meta_p, 3),
        "image_score": round(image_p, 3),
        "fused_score": round(fused, 3),
        "weights": weights,
        "verdict": verdict(fused),
    }
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="GitHub repository URL")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    print(json.dumps(predict_repo(args.url, args.token), indent=2))
