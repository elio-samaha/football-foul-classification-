#!/usr/bin/env python3
"""
Utility script to download the SoccerNet-MVFoul dataset.

This is a light wrapper around the official SoccerNet API as documented in:
https://github.com/SoccerNet/sn-mvfoul

Usage
-----
You must first obtain an NDA password from SoccerNet. Then run:

    python scripts/download_mvfoul.py --password YOUR_PASSWORD

By default this downloads all splits (train/valid/test/challenge) into
`data/SoccerNet`. You can change the output directory and splits via CLI
flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SoccerNet-MVFoul via the SoccerNet API.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("data/SoccerNet"),
        help="Root directory where SoccerNet data will be downloaded.",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="NDA password for SoccerNet. If omitted, the SOCCERNET_PASSWORD env var is used.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test,challenge",
        help="Comma-separated list of splits to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from SoccerNet.Downloader import SoccerNetDownloader as SNdl
    except Exception as exc:  # pragma: no cover - thin wrapper
        raise SystemExit(
            "Failed to import SoccerNet. Install it with `pip install SoccerNet` "
            "and ensure the package is available in your environment."
        ) from exc

    password = args.password
    if password is None:
        import os

        password = os.getenv("SOCCERNET_PASSWORD")
    if not password:
        raise SystemExit(
            "No password provided. Either pass --password or set the "
            "SOCCERNET_PASSWORD environment variable."
        )

    local_dir: Path = args.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    splits: List[str] = [s.strip() for s in args.splits.split(",") if s.strip()]

    downloader = SNdl(LocalDirectory=str(local_dir))
    downloader.downloadDataTask(task="mvfouls", split=splits, password=password)

    print(f"Downloaded SoccerNet-MVFoul splits {splits} into {local_dir}")


if __name__ == "__main__":
    main()

