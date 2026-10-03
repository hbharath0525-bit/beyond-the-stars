"""Central configuration for Beyond the Stars."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ---------- Multimodal fusion weights ----------
FUSION_WEIGHTS = {"text": 0.50, "meta": 0.30, "image": 0.20}

# Scores below this are flagged as scams
SCAM_THRESHOLD = 0.55

# ---------- Keyword lexicons ----------
SCAM_KEYWORDS = [
    "airdrop", "free money", "double your", "100x", "guaranteed profit",
    "pump and dump", "send eth", "send btc", "seed phrase", "private key",
    "recovery phrase", "wallet drain", "presale", "elon musk giveaway",
    "whatsapp", "telegram investment", "binary options", "passive income guarantee",
    "no risk", "instant withdrawal", "hack tool", "cracker", "mod apk premium free",
    "cheat undetected", "free robux", "free vbucks", "gift card generator",
    "password decryptor", "ransomware as a service",
]

SUSPICIOUS_REPO_NAME_PATTERNS = [
    "hack", "cracker", "bypass", "premium-free", "mod-apk", "generator",
    "giveaway", "airdrop", "free-", "-free", "unlock", "cheat",
]
