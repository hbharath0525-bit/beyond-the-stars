"""Step 1 - Build a labelled dataset of GitHub repos (scam vs legitimate)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import pandas as pd
from src.github_client import GitHubRepoMiner


def collect(urls: list[str], label: int, token: str | None, out: str):
    miner = GitHubRepoMiner(token)
    rows = []
    for url in urls:
        try:
            mined = miner.mine(url)
            rows.append({
                "url": url, "label": label,
                "readme": mined["readme"],
                "metadata": json.dumps(mined["metadata"]),
                "n_images": len(mined["images"]),
            })
            print(f"[OK] {url}")
        except Exception as e:
            print(f"[FAIL] {url} -> {e}")
    df = pd.DataFrame(rows)
    header = not pd.io.common.file_exists(out)
    df.to_csv(out, mode="a", index=False, header=header)
    print(f"Saved {len(df)} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-file", required=True, help="text file, one repo URL per line")
    ap.add_argument("--label", type=int, required=True, help="1 = scam, 0 = legitimate")
    ap.add_argument("--out", default="data/repos.csv")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    urls = [l.strip() for l in open(args.urls_file) if l.strip()]
    collect(urls, args.label, args.token, args.out)
