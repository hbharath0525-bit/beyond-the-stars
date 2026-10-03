"""Modality 1: NLP features from README + description."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

from config.config import SCAM_KEYWORDS


def clean_markdown(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)   # images
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)     # links
    text = re.sub(r"[#>*_`~|-]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def scam_keyword_score(text: str) -> float:
    """Fraction of scam keywords present (0..1)."""
    t = text.lower()
    hits = sum(1 for kw in SCAM_KEYWORDS if kw in t)
    return hits / len(SCAM_KEYWORDS)


class TextScamModel:
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                   min_df=2, stop_words="english")
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    def train(self, texts, labels):
        X = self.vec.fit_transform([clean_markdown(t) for t in texts])
        kw = np.array([scam_keyword_score(t) for t in texts]).reshape(-1, 1)
        self.clf.fit(np.hstack([X.toarray(), kw]), labels)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        X = self.vec.transform([clean_markdown(t) for t in texts])
        kw = np.array([scam_keyword_score(t) for t in texts]).reshape(-1, 1)
        return self.clf.predict_proba(np.hstack([X.toarray(), kw]))[:, 1]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
