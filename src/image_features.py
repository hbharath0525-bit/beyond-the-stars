"""Modality 3: image analysis (OCR text risk + perceptual-hash similarity to known scam screenshots).

Uses pytesseract if the Tesseract binary is installed; otherwise falls back
gracefully to a keyword-free hash-only model.
"""
import io
import numpy as np
from PIL import Image
import joblib

from config.config import SCAM_KEYWORDS


def _avg_hash(img: Image.Image) -> str:
    try:
        import imagehash
        return str(imagehash.average_hash(img.convert("L").resize((16, 16))))
    except ImportError:
        # Fallback: simple 16x16 grayscale bit hash
        g = img.convert("L").resize((16, 16))
        px = list(g.getdata())
        avg = sum(px) / len(px)
        bits = "".join("1" if v >= avg else "0" for v in px)
        return f"{int(bits, 2):x}"


def _ocr_text(img: Image.Image) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def ocr_risk_score(text: str) -> float:
    t = text.lower()
    hits = sum(1 for kw in SCAM_KEYWORDS if kw in t)
    # common scam-image phrases
    extra = ["congratulations", "you won", "claim now", "verify wallet",
             "connect wallet", "limited time", "act now"]
    hits += sum(1 for kw in extra if kw in t)
    return min(hits / 6.0, 1.0)


class ImageScamModel:
    """Lightweight image model: similarity to a bank of known scam screenshots."""

    def __init__(self, hash_size: int = 16):
        self.hash_size = hash_size
        self.scam_hashes: list[str] = []
        self.ocr_weight = 0.5

    def _hash_to_bits(self, h: str) -> np.ndarray:
        return np.array([int(b) for b in bin(int(h, 16))[2:].zfill(self.hash_size ** 2)])

    def fit(self, scam_images: list[bytes], clean_images: list[bytes] | None = None):
        self.scam_hashes = [_avg_hash(Image.open(io.BytesIO(b))) for b in scam_images]
        self.clean_hashes = [_avg_hash(Image.open(io.BytesIO(b))) for b in (clean_images or [])]
        return self

    def predict_proba(self, images: list[bytes]) -> np.ndarray:
        scores = []
        scam_bank = [self._hash_to_bits(h) for h in getattr(self, "scam_hashes", [])]
        clean_bank = [self._hash_to_bits(h) for h in getattr(self, "clean_hashes", [])]
        for blob in images:
            img = Image.open(io.BytesIO(blob))
            bits = self._hash_to_bits(_avg_hash(img))
            sim_scam = max((self._hamming_sim(bits, b) for b in scam_bank), default=0.0)
            sim_clean = max((self._hamming_sim(bits, b) for b in clean_bank), default=0.0)
            ocr = ocr_risk_score(_ocr_text(img))
            score = 0.35 * sim_scam + 0.15 * (1 - sim_clean) + self.ocr_weight * ocr
            scores.append(min(score, 1.0))
        return np.array(scores) if scores else np.array([0.0])

    @staticmethod
    def _hamming_sim(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - float(np.mean(a != b))

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
