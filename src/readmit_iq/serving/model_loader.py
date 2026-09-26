"""Verify trusted provenance and exact bytes before unpickling the frozen pipeline once."""

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import joblib
from threadpoolctl import threadpool_limits

from readmit_iq.serving.contract import CONTRACT, FEATURES, bundle_root
from readmit_iq.serving.errors import ModelIntegrityError

logger = logging.getLogger("readmit_iq.serving")


def immutable(value):
    if isinstance(value, dict):
        return MappingProxyType({k: immutable(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(immutable(v) for v in value)
    return value


def trusted_bytes(path, expected, code="bundle_integrity_failed"):
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ModelIntegrityError(
            code, "Required serving bundle files are missing or unreadable."
        ) from error
    if hashlib.sha256(data).hexdigest() != expected:
        raise ModelIntegrityError(code, "Serving bundle integrity verification failed.")
    return data


@dataclass(frozen=True)
class FrozenModel:
    _pipeline: object = field(repr=False)
    _metadata_json: str = field(repr=False)
    _guard: threading.Lock = field(default_factory=threading.Lock, repr=False)
    artifact_verified: bool = field(default=True, init=False)

    @property
    def metadata(self):
        return immutable(json.loads(self._metadata_json))

    def predict_proba(self, frame):
        if tuple(frame.columns) != FEATURES:
            raise ValueError("Frozen model requires the exact ordered feature schema")
        # Bound native CPU work and avoid concurrent threadpool-limit changes.
        with self._guard, threadpool_limits(limits=2):
            return self._pipeline.predict_proba(frame)


def load_frozen_model(root: Path | None = None, artifact: Path | None = None) -> FrozenModel:
    """No fitting, fallback artifact, model download, Git command or dataset access."""
    root = root or bundle_root()
    checked = {}
    for relative, expected in CONTRACT["files"].items():
        checked[relative] = trusted_bytes(root / relative, expected)
        if relative.startswith("src/") and relative.endswith(".py"):
            module = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
            found = importlib.util.find_spec(module)
            if found is None or not found.origin:
                raise ModelIntegrityError(
                    "runtime_incompatible", "Required frozen pipeline code is unavailable."
                )
            trusted_bytes(Path(found.origin), expected, "runtime_incompatible")
    spec = json.loads(checked["reports/modeling/final_model_specification.json"])
    metadata = json.loads(checked["reports/modeling/final_model_metadata.json"])
    primary = spec["models"][spec["primary"]]
    if (
        spec["version"] != CONTRACT["model_version"]
        or metadata["version"] != spec["version"]
        or metadata["primary"] != CONTRACT["primary"]
        or tuple(metadata["features"]) != FEATURES
        or primary["numeric_features"] + primary["categorical_features"] != list(FEATURES)
        or metadata["pipeline_sha256"] != primary["sha256"]
        or metadata["targeting"] != spec["targeting"]
    ):
        raise ModelIntegrityError(
            "metadata_incompatible", "Frozen model metadata or schema is incompatible."
        )
    if sys.version_info[:2] != (3, 12):
        raise ModelIntegrityError("runtime_incompatible", "Frozen serving requires Python 3.12.")
    for package in CONTRACT["runtime_packages"]:
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != spec["library_versions"][package]:
            raise ModelIntegrityError(
                "runtime_incompatible", "Install the pinned serving runtime dependencies."
            )
    path = artifact or Path(
        os.environ.get("READMITIQ_MODEL_PATH", root / metadata["inference_pipeline"])
    )
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ModelIntegrityError(
            "model_missing", "Restore the trusted frozen artifact, then run make verify-model."
        ) from error
    if size != primary["artifact_bytes"]:
        raise ModelIntegrityError(
            "model_hash_mismatch", "Frozen artifact integrity verification failed."
        )
    data = trusted_bytes(path, primary["sha256"], "model_hash_mismatch")
    try:
        # Unpickle the same verified buffer, not a reopened path (avoids replacement races).
        pipeline = joblib.load(io.BytesIO(data))
        if (
            type(pipeline[-1]).__name__ != primary["estimator_class"]
            or pipeline[-1].get_params(deep=False) != primary["exact_estimator_parameters"]
            or pipeline.named_steps["features"].configuration != "raw"
            or pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
            != primary["preprocessing"]["output_columns"]
        ):
            raise ValueError("Estimator or preprocessing contract mismatch")
    except Exception as error:
        raise ModelIntegrityError(
            "model_incompatible", "Frozen artifact cannot be loaded by this runtime."
        ) from error
    logger.info("model_verified", extra={"model_version": metadata["version"]})
    return FrozenModel(pipeline, json.dumps(metadata))


if __name__ == "__main__":
    try:
        model = load_frozen_model()
        print(
            json.dumps(
                {
                    "artifact_verified": True,
                    "version": model.metadata["version"],
                    "sha256": model.metadata["pipeline_sha256"],
                },
                indent=2,
            )
        )
    except ModelIntegrityError as error:
        print(f"{error.code}: {error.message}", file=sys.stderr)
        raise SystemExit(1) from None
