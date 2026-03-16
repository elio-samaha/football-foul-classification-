#!/usr/bin/env python3
"""Download the SoccerNet MVFoul dataset.

Requires a password obtained by signing the SoccerNet NDA:
https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform

Usage:
    python scripts/download_mvfoul.py --password YOUR_PASSWORD [--output_dir data/mvfoul] [--version 224p]
"""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SoccerNet MVFoul dataset")
    parser.add_argument("--password", required=True, help="SoccerNet NDA password")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/mvfoul",
        help="Local directory to store the dataset",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test", "challenge"],
        help="Splits to download",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help='Video resolution version (e.g. "720p"). Default downloads standard resolution.',
    )
    args = parser.parse_args()

    from SoccerNet.Downloader import SoccerNetDownloader as SNdl

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    downloader = SNdl(LocalDirectory=str(out))
    dl_kwargs: dict = dict(
        task="mvfouls",
        split=args.splits,
        password=args.password,
    )
    if args.version:
        dl_kwargs["version"] = args.version

    print(f"Downloading MVFoul splits {args.splits} to {out} ...")
    downloader.downloadDataTask(**dl_kwargs)
    print("Download complete. Unzip each split folder (Train, Valid, Test, Chall) in-place.")
    print("Expected structure after unzip:")
    print(f"  {out}/Train/action_XXXX/clip_N.mp4")
    print(f"  {out}/Valid/action_XXXX/clip_N.mp4")
    print(f"  {out}/Test/action_XXXX/clip_N.mp4")


if __name__ == "__main__":
    main()
