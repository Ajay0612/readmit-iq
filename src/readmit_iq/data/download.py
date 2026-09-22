"""Download and verify the original UCI release; never silently replace changed data."""

import hashlib
import json
import logging
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests

from readmit_iq.config import configure_logging, load_config

LOGGER = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Checksum mismatch for {path.name}; review source before proceeding")


def fetch(url: str, destination: Path) -> None:
    """Stream to a temporary sibling then atomically rename a complete download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temporary = Path(temp.name)
        try:
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    temp.write(chunk)
            temp.close()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def extract_verified(archive: Path, destination: Path, checksums: dict[str, str]) -> None:
    """Extract only the known flat filenames, checking bytes before installation."""
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist()) != set(checksums):
            raise ValueError("Archive contents differ from the expected UCI release")
        for name, expected in checksums.items():
            if Path(name).name != name:
                raise ValueError("Expected archive member must be a flat filename")
            target = destination / name
            if target.exists():
                verify_file(target, expected)
                continue
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as temp:
                temporary = Path(temp.name)
                try:
                    with bundle.open(name) as source:
                        shutil.copyfileobj(source, temp)
                    temp.close()
                    verify_file(temporary, expected)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)


def main() -> None:
    configure_logging()
    config = load_config()
    raw = config.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "uci_diabetes_296.zip"
    if not archive.exists():
        LOGGER.info("Downloading original UCI archive")
        fetch(config.dataset.archive_url, archive)
    verify_file(archive, config.dataset.archive_sha256)
    extract_verified(archive, raw, config.dataset.file_sha256)
    metadata = raw / "uci_metadata.json"
    if not metadata.exists():
        fetch(config.dataset.metadata_url, metadata)
    payload = json.loads(metadata.read_text())
    if payload.get("status") != 200 or payload["data"]["uci_id"] != 296:
        raise ValueError("Unexpected UCI metadata response")
    manifest_path = raw / "manifest.json"
    # Preserve the time of first successful local verification across repeat runs.
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("first_verified_utc", datetime.now(UTC).isoformat())
    manifest.update(
        {
            "source_url": config.dataset.source_url,
            "archive_url": config.dataset.archive_url,
            "metadata_url": config.dataset.metadata_url,
            "license": "CC BY 4.0",
            "files": {
                path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                for path in [archive, metadata, *(raw / n for n in config.dataset.file_sha256)]
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    config.report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, config.report_dir / "source_manifest.json")
    LOGGER.info("Verified archive and both source CSV files; raw data left unchanged")


if __name__ == "__main__":
    main()
