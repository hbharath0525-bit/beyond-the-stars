"""Generate a synthetic demo dataset so the project runs without a GitHub token.

In real use: collect real repos via `python scripts/collect_data.py`.
"""
import json
import pandas as pd

SCAM_READMES = [
    "# FREE ETHEREUM AIRDROP 🚀\n\nSend 0.1 ETH to 0xABC... and receive 0.2 ETH back instantly! Guaranteed profit, no risk. Limited time only!\n\n## How to join\n1. Send ETH\n2. Double your money\n3. Withdraw instantly\n\nContact us on Telegram @scam",
    "# Wallet Drainer Pro\n\nConnect your wallet and verify your seed phrase to claim free NFT drops. 100x gains on presale tokens!\n\n- airdrop hunting tool\n- passive income guarantee\n- no risk investment",
    "# Mod APK Premium Free\n\nUnlock all premium features for free! No root needed. hack tool v2.1, undetected cracker.\n\nDownload now - 100% working",
    "# Elon Musk Crypto Giveaway\n\nElon is giving away 5000 BTC! Send 1 BTC get 2 back. Double your crypto today. Instant withdrawal guaranteed.",
]
LEGIT_READMES = [
    "# fastapi-cache\n\nA lightweight caching layer for FastAPI built on Redis. Supports TTL, async, and type hints.\n\n```pip install fastapi-cache```\n\n## Usage\nSee docs/ for examples and benchmarks.",
    "# numpy-financial\n\nFinancial functions for NumPy: IRR, NPV, loan amortization. Well tested with pytest, CI on GitHub Actions.",
    "# react-grid-layout\n\nA grid layout system for React with draggable and resizable widgets. Used by Grafana. MIT licensed, contributions welcome.",
    "# transformer-lm\n\nMinimal PyTorch implementation of a decoder-only transformer language model with full training and eval scripts, Weights & Biases logging.",
]

SCAM_META = [
    {"full_name": "crypto-elon/airdrop-doubler", "description": "double your eth free money airdrop", "stars": 12, "forks": 0, "watchers": 3, "open_issues": 8, "size_kb": 45, "age_days": 3, "has_pages": 1, "fork": 0, "archived": 0, "num_topics": 1, "license_present": 0},
    {"full_name": "nft-tools/wallet-drain", "description": "connect wallet claim airdrop", "stars": 5, "forks": 1, "watchers": 1, "open_issues": 2, "size_kb": 12, "age_days": 1, "has_pages": 1, "fork": 0, "archived": 0, "num_topics": 0, "license_present": 0},
    {"full_name": "modz/premium-free-apk", "description": "mod apk premium free unlock", "stars": 40, "forks": 2, "watchers": 6, "open_issues": 1, "size_kb": 900, "age_days": 7, "has_pages": 0, "fork": 0, "archived": 0, "num_topics": 0, "license_present": 0},
    {"full_name": "btc-giveaway/free-btc", "description": "send btc get double back guaranteed profit", "stars": 3, "forks": 0, "watchers": 2, "open_issues": 0, "size_kb": 8, "age_days": 2, "has_pages": 1, "fork": 1, "archived": 0, "num_topics": 0, "license_present": 0},
]
LEGIT_META = [
    {"full_name": "dev/fastapi-cache", "description": "Redis caching layer for FastAPI", "stars": 2450, "forks": 180, "watchers": 45, "open_issues": 22, "size_kb": 340, "age_days": 620, "has_pages": 0, "fork": 0, "archived": 0, "num_topics": 4, "license_present": 1},
    {"full_name": "quants/numpy-financial", "description": "financial functions for numpy", "stars": 890, "forks": 96, "watchers": 30, "open_issues": 5, "size_kb": 210, "age_days": 1100, "has_pages": 0, "fork": 0, "archived": 0, "num_topics": 3, "license_present": 1},
    {"full_name": "ui-libs/react-grid-layout", "description": "draggable grid layout for react", "stars": 18200, "forks": 2400, "watchers": 310, "open_issues": 120, "size_kb": 5200, "age_days": 2900, "has_pages": 1, "fork": 0, "archived": 0, "num_topics": 6, "license_present": 1},
    {"full_name": "ml-research/transformer-lm", "description": "minimal transformer LM in pytorch", "stars": 3200, "forks": 410, "watchers": 88, "open_issues": 14, "size_kb": 1400, "age_days": 540, "has_pages": 0, "fork": 0, "archived": 0, "num_topics": 5, "license_present": 1},
]

rows = []
for i in range(len(SCAM_READMES)):
    rows.append({"url": f"https://github.com/{SCAM_META[i]['full_name']}", "label": 1,
                 "readme": SCAM_READMES[i], "metadata": json.dumps(SCAM_META[i]), "n_images": 1})
    rows.append({"url": f"https://github.com/{LEGIT_META[i]['full_name']}", "label": 0,
                 "readme": LEGIT_READMES[i], "metadata": json.dumps(LEGIT_META[i]), "n_images": 0})

pd.DataFrame(rows).to_csv("data/repos.csv", index=False)
print("Demo dataset written:", len(rows), "rows")
