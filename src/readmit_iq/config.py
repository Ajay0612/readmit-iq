"""Validated YAML settings with paths relative to the configuration's project root."""

import logging
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    source_url: str
    archive_url: str
    archive_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    metadata_url: str
    expected_rows: int
    expected_columns: list[str]
    file_sha256: dict[str, str]


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    random_seed: int = Field(ge=0)
    raw_dir: Path
    report_dir: Path
    dataset: DatasetConfig
    numeric_bounds: dict[str, tuple[int, int | None]]


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Resolve paths from configs/.., independently of the caller's working directory."""
    default = Path.cwd() / "configs/config.yaml"
    config_path = Path(path or os.getenv("READMITIQ_CONFIG", default)).resolve()
    config = ProjectConfig.model_validate(yaml.safe_load(config_path.read_text()))
    root = config_path.parent.parent
    config.raw_dir = (root / config.raw_dir).resolve()
    config.report_dir = (root / config.report_dir).resolve()
    return config


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("READMITIQ_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
