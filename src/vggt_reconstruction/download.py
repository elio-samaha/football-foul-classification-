from __future__ import annotations

from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

from SoccerNet.Downloader import SoccerNetDownloader


def download_mvfoul_dataset(
    local_directory: Path,
    password: str,
    splits: Iterable[str] = ("train", "valid", "test"),
    version: str | None = None,
    extract: bool = True,
    task: str = "mvfouls",
) -> Path:
    """Download MVFoul into the standard SoccerNet task directory."""

    local_directory = Path(local_directory)
    local_directory.mkdir(parents=True, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(local_directory))
    downloader.downloadDataTask(
        task=task,
        split=list(splits),
        password=password,
        version=version,
    )

    dataset_root = local_directory / task
    if extract:
        extract_archives(dataset_root)
    return dataset_root


def extract_archives(dataset_root: Path) -> None:
    """Extract all zip archives produced by the SoccerNet downloader."""

    dataset_root = Path(dataset_root)
    for archive_path in sorted(dataset_root.glob("*.zip")):
        with ZipFile(archive_path) as archive:
            archive.extractall(dataset_root)
