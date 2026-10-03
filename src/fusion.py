"""Weighted late-fusion of the three modality models."""
import numpy as np
from config.config import FUSION_WEIGHTS, SCAM_THRESHOLD


def fuse(text_p: np.ndarray, meta_p: np.ndarray, image_p: np.ndarray,
         weights: dict | None = None) -> np.ndarray:
    w = weights or FUSION_WEIGHTS
    return (w["text"] * np.asarray(text_p) +
            w["meta"] * np.asarray(meta_p) +
            w["image"] * np.asarray(image_p))


def verdict(fused_score: float, threshold: float = SCAM_THRESHOLD) -> str:
    if fused_score >= threshold:
        return "⚠️ SCAM (high confidence)" if fused_score >= threshold + 0.2 else "⚠️ LIKELY SCAM"
    if fused_score >= threshold - 0.15:
        return "⚡ SUSPICIOUS"
    return "✅ LIKELY LEGITIMATE"
