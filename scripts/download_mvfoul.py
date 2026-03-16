#!/usr/bin/env python3
"""Download the SoccerNet MVFoul dataset.

Two modes are supported:
  1. HuggingFace Hub (default) -- no password required for metadata, but the
     video archives are gated and still require an NDA.
  2. SoccerNet Downloader API  -- requires the NDA password.

Usage examples:

    # Via HuggingFace Hub (recommended)
    python scripts/download_mvfoul.py --dest data/mvfoul

    # Via SoccerNet Downloader
    python scripts/download_mvfoul.py --dest data/mvfoul --backend soccernet --password <NDA_PASSWORD>

After downloading, unzip any .zip archives while preserving the folder names
(Train, Valid, Test, Chall).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def download_huggingface(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading SoccerNet/SN-MVFouls-2025 to {dest} via HuggingFace Hub ...")
    snapshot_download(
        repo_id="SoccerNet/SN-MVFouls-2025",
        repo_type="dataset",
        revision="main",
        local_dir=str(dest),
    )
    print("Download complete.")


def download_soccernet(dest: Path, password: str) -> None:
    from SoccerNet.Downloader import SoccerNetDownloader as SNdl

    print(f"Downloading mvfouls task to {dest} via SoccerNet API ...")
    downloader = SNdl(LocalDirectory=str(dest))
    downloader.downloadDataTask(
        task="mvfouls",
        split=["train", "valid", "test", "challenge"],
        password=password,
    )
    print("Download complete.")


def unzip_all(dest: Path) -> None:
    """Unzip every .zip found under *dest* and remove the archive."""
    for zf in sorted(dest.rglob("*.zip")):
        target = zf.parent / zf.stem
        print(f"Extracting {zf} -> {target} ...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(target)
        zf.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the MVFoul dataset")
    parser.add_argument("--dest", type=str, default="data/mvfoul", help="Local destination directory")
    parser.add_argument(
        "--backend",
        choices=["huggingface", "soccernet"],
        default="huggingface",
        help="Download backend (default: huggingface)",
    )
    parser.add_argument("--password", type=str, default=None, help="NDA password (soccernet backend only)")
    parser.add_argument("--unzip", action="store_true", help="Auto-unzip downloaded archives")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.backend == "huggingface":
        download_huggingface(dest)
    else:
        if not args.password:
            parser.error("--password is required when using the soccernet backend")
        download_soccernet(dest, args.password)

    if args.unzip:
        unzip_all(dest)


if __name__ == "__main__":
    main()
