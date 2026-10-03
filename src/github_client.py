"""GitHub API client: pulls repo metadata, README, and images for analysis."""
import os
import re
import base64
import requests


def _owner_repo(url: str):
    m = re.search(r"github\.com[:/](?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)", url)
    if not m:
        raise ValueError(f"Not a valid GitHub URL: {url}")
    return m.group("owner"), m.group("repo")


class GitHubRepoMiner:
    """Collects all three modalities from a single repository."""

    def __init__(self, token: str | None = None):
        from github import Github
        token = token or os.getenv("GITHUB_TOKEN")
        self.gh = Github(token) if token else Github()
        self.sess = requests.Session()
        if token:
            self.sess.headers["Authorization"] = f"Bearer {token}"

    def mine(self, url: str, max_images: int = 5) -> dict:
        owner, name = _owner_repo(url)
        repo = self.gh.get_repo(f"{owner}/{name}")

        # ---------- Modality 1: metadata ----------
        created = repo.created_at
        pushed = repo.pushed_at
        metadata = {
            "full_name": repo.full_name,
            "description": repo.description or "",
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "watchers": repo.watchers_count,
            "open_issues": repo.open_issues_count,
            "size_kb": repo.size,
            "age_days": (pushed - created).days if pushed and created else 0,
            "has_pages": int(bool(repo.get_topics() is not None)),
            "fork": int(repo.fork),
            "archived": int(repo.archived),
            "num_topics": len(repo.get_topics()),
            "license_present": int(repo.license is not None),
            "default_branch": repo.default_branch,
        }

        # ---------- Modality 2: text (README + description) ----------
        try:
            readme = repo.get_readme().decoded_content.decode("utf-8", errors="ignore")
        except Exception:
            readme = ""

        images = self._fetch_images(repo, max_images)
        return {"metadata": metadata, "readme": readme, "images": images, "url": url}

    def _fetch_images(self, repo, max_images: int) -> list[bytes]:
        """Grab image blobs from the repo root/docs folder."""
        imgs = []
        exts = (".png", ".jpg", ".jpeg", ".webp")
        try:
            tree = repo.get_git_tree(repo.default_branch, recursive=True)
            for item in tree.tree:
                if item.type == "blob" and item.path.lower().endswith(exts):
                    blob = repo.get_git_blob(item.sha)
                    imgs.append(base64.b64decode(blob.content))
                    if len(imgs) >= max_images:
                        break
        except Exception:
            pass
        return imgs
