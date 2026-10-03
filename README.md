# 🌟 Beyond the Stars: Multimodal Detection of Scams on GitHub

Detects scam / phishing repositories on GitHub by **fusing three modalities**:

| Modality | Source | Model |
|---|---|---|
| 📝 Text | README + description (markdown-cleaned, TF-IDF + scam-keyword score) | Logistic Regression |
| 📊 Metadata | stars, forks, age, license, topics, name-risk lexicon... | Gradient Boosting |
| 🖼️ Images | repo screenshots via OCR (Tesseract) + perceptual hash (imagehash) | Hash-bank similarity |

A weighted late-fusion layer combines all three scores into a final 0–1 scam probability with a verdict: `✅ LIKELY LEGITIMATE / ⚡ SUSPICIOUS / ⚠️ LIKELY SCAM / ⚠️ SCAM`.

## 🗂️ Project structure
```
beyond_the_stars/
├── config/config.py            # fusion weights, lexicons, paths
├── src/
│   ├── github_client.py        # GitHub API miner (PyGithub)
│   ├── text_features.py        # Modality 1: text
│   ├── meta_features.py        # Modality 2: metadata
│   ├── image_features.py       # Modality 3: images (OCR + pHash)
│   └── fusion.py               # weighted fusion + verdicts
├── scripts/
│   ├── collect_data.py         # Step 1: build dataset from real repos
│   ├── make_demo_data.py       # synthetic demo dataset (no token needed)
│   ├── train.py                # Step 2: train all 3 models + learn weights
│   └── predict.py              # Step 3: score a live repo URL
└── data/, models/, reports/
```

## 🚀 Quick start (VS Code)

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) free GitHub token -> higher API rate limits
#    create .env:  GITHUB_TOKEN=ghp_xxxx

# 4a. RUN THE OFFLINE DEMO (no token, no network):
python scripts/demo_offline.py

# 4b. OR: end-to-end with real repos
echo "https://github.com/some/scam-repo" > scam_urls.txt
echo "https://github.com/torvalds/linux" > legit_urls.txt
python scripts/collect_data.py --urls-file scam_urls.txt --label 1 --out data/repos.csv
python scripts/collect_data.py --urls-file legit_urls.txt --label 0 --out data/repos.csv
python scripts/train.py --data data/repos.csv

# 5. Predict a repo
python scripts/predict.py https://github.com/some/repo
```

## 🧠 How the fusion works
Each modality model outputs P(scam). The final score is
`0.5·text + 0.3·meta + 0.2·image` (weights re-learned on validation data in `train.py`).
Scores ≥ 0.55 → scam flagged; ≥ 0.40 → suspicious.

## 🖼️ Image modality notes
- OCR requires the Tesseract binary (`apt install tesseract-ocr` / `brew install tesseract`); the code degrades gracefully without it.
- To use the image bank, extract screenshots of known scam repos into `data/images/scam/` and retrain — images hash-matched within 10% Hamming distance raise the image score.
