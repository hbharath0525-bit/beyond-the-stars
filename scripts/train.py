"""Step 2 - Train all three modality models + save fusion weights."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import pandas as pd

from config.config import MODEL_DIR
from src.text_features import TextScamModel
from src.meta_features import MetaScamModel
from src.image_features import ImageScamModel
from src.fusion import fuse


def main(data_csv: str):
    df = pd.read_csv(data_csv)
    labels = df["label"].values
    metadata = [json.loads(m) for m in df["metadata"]]

    # --- Text model ---
    text_model = TextScamModel().train(df["readme"].fillna(""), labels)
    text_p = text_model.predict_proba(df["readme"].fillna(""))

    # --- Metadata model ---
    meta_model = MetaScamModel().train(metadata, labels)
    meta_p = meta_model.predict_proba(metadata)

    # --- Image model (self-supervised: 'scam bank' = images of scam repos) ---
    img_model = ImageScamModel().fit(scam_images=[], clean_images=[])  # populate from data/images if available
    image_p = np.zeros(len(df))

    # --- Learn fusion weights on validation split (simple grid search) ---
    best_w, best_acc = {"text": .5, "meta": .3, "image": .2}, 0.0
    for tw in np.arange(0.2, 0.8, 0.1):
        for mw in np.arange(0.1, 0.6, 0.1):
            iw = 1 - tw - mw
            if iw < 0:
                continue
            fused = fuse(text_p, meta_p, image_p, {"text": tw, "meta": mw, "image": iw})
            acc = ((fused >= 0.5).astype(int) == labels).mean()
            if acc > best_acc:
                best_acc, best_w = acc, {"text": tw, "meta": mw, "image": iw}

    fused = fuse(text_p, meta_p, image_p, best_w)
    pred = (fused >= 0.5).astype(int)
    acc = (pred == labels).mean()

    text_model.save(MODEL_DIR / "text_model.joblib")
    meta_model.save(MODEL_DIR / "meta_model.joblib")
    img_model.save(MODEL_DIR / "image_model.joblib")
    json.dump({"weights": best_w, "train_acc": float(acc)},
              open(MODEL_DIR / "fusion.json", "w"), indent=2)

    print(f"Train accuracy: {acc:.3f}")
    print(f"Learned fusion weights: {best_w}")
    print(f"Top metadata features:\n{meta_model.importance().head(8)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/repos.csv")
    main(ap.parse_args().data)
