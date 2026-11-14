"""Run the unchanged XGBoost release with its published dependency lock (Mac/Linux)."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPOSITORY = "https://github.com/Query-farm/vgi-xgboost.git"
RELEASES = {"v0.1.0": "eb4f1991e788801aa11d58e231f019f0cf613f44"}


def hashes(source: Path) -> dict[str, str]:
    files = [
        source / "xgboost_worker.py",
        source / "pyproject.toml",
        source / "uv.lock",
        *sorted((source / "vgi_xgboost").rglob("*.py")),
    ]
    return {str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def checkout(release: str) -> Path:
    revision = RELEASES[release]
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vgi-xgboost-releases"
    root.mkdir(parents=True, exist_ok=True)
    target = root / revision
    with (root / (revision + ".lock")).open("a") as lock:
        deadline = time.monotonic() + 240
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Another XGBoost release installation did not finish"
                    ) from None
                time.sleep(0.1)
        if target.exists():
            marker = json.loads((target / "verified.json").read_text())
            if marker["revision"] != revision or marker["sha256"] != hashes(target / "source"):
                raise RuntimeError(
                    "Cached release failed verification; remove this release cache and retry"
                )
            return target / "source"
        with tempfile.TemporaryDirectory(dir=root, prefix="install-") as temporary:
            stage = Path(temporary) / "ready"
            stage.mkdir()
            source = stage / "source"
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", release, REPOSITORY, str(source)],
                check=True,
                stdout=sys.stderr,
            )
            actual = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != revision:
                raise RuntimeError(
                    "Public release tag no longer matches its reviewed immutable revision"
                )
            (stage / "verified.json").write_text(
                json.dumps({"revision": revision, "sha256": hashes(source)})
            )
            stage.rename(target)
    return target / "source"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", choices=RELEASES)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = checkout(args.release)
    # Explicit python avoids interpreting the root script's looser PEP 723 metadata.
    # uv serializes environment sync; setup messages go to stderr, leaving Arrow stdout intact.
    subprocess.run(
        [
            "uv",
            "sync",
            "--frozen",
            "--no-dev",
            "--no-install-project",
            "--python",
            "3.13",
            "--project",
            str(source),
        ],
        check=True,
        stdout=sys.stderr,
    )
    python = str(source / ".venv" / "bin" / "python")
    os.execv(python, [python, str(source / "xgboost_worker.py"), *args.worker_args])


if __name__ == "__main__":
    main()
