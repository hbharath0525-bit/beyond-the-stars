"""Modality 2: repository metadata / behavioral features."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from config.config import SUSPICIOUS_REPO_NAME_PATTERNS

META_COLS = ["stars", "forks", "watchers", "open_issues", "size_kb",
             "age_days", "has_pages", "fork", "archived", "num_topics",
             "license_present"]


def name_risk_score(full_name: str) -> float:
    n = full_name.lower()
    hits = sum(1 for p in SUSPICIOUS_REPO_NAME_PATTERNS if p in n)
    return min(hits / 3.0, 1.0)


def desc_keyword_hits(description: str) -> float:
    from config.config import SCAM_KEYWORDS
    d = (description or "").lower()
    return min(sum(1 for kw in SCAM_KEYWORDS if kw in d) / 4.0, 1.0)


def build_meta_frame(metadata_list: list[dict]) -> pd.DataFrame:
    rows = []
    for m in metadata_list:
        row = {c: m.get(c, 0) for c in META_COLS}
        row["name_risk"] = name_risk_score(m.get("full_name", ""))
        row["desc_risk"] = desc_keyword_hits(m.get("description", ""))
        # derived signals: engagement ratios
        row["fork_ratio"] = row["forks"] / (row["stars"] + 1)
        row["issue_ratio"] = row["open_issues"] / (row["stars"] + 1)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


class MetaScamModel:
    def __init__(self):
        self.clf = GradientBoostingClassifier(n_estimators=250, max_depth=3,
                                              learning_rate=0.05, random_state=42)

    def train(self, metadata_list, labels):
        X = build_meta_frame(metadata_list)
        self.clf.fit(X, labels)
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, metadata_list) -> np.ndarray:
        X = build_meta_frame(metadata_list)
        return self.clf.predict_proba(X[self.feature_names_])[:, 1]

    def importance(self) -> pd.Series:
        return pd.Series(self.clf.feature_importances_,
                         index=self.feature_names_).sort_values(ascending=False)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
