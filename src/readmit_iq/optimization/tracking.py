"""Explicit local-only MLflow tracking; no server, credentials, autologging or raw data upload."""

import json
import os
import time
from pathlib import Path

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.entities import Metric, Param  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402


class LocalTracker:
    def __init__(self, root: Path, policy: dict, revision: str):
        directory = (root / policy["tracking_dir"]).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("MLflow must remain inside the local project")
        directory.mkdir(parents=True, exist_ok=True)
        self.uri = "sqlite:///" + str(directory / "tracking.sqlite")
        self.client = MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)
        experiment = self.client.get_experiment_by_name(policy["experiment_name"])
        self.experiment_id = (
            experiment.experiment_id
            if experiment
            else self.client.create_experiment(
                policy["experiment_name"], artifact_location=(directory / "artifacts").as_uri()
            )
        )
        self.tags = {"phase": "4", "git_commit": revision, "seed": str(policy["random_seed"])}

    def log(
        self, name: str, parameters: dict, metrics: dict, artifacts: list[Path] | None = None
    ) -> str:
        """Track scalar summaries and explicitly provided aggregate files only."""
        run = self.client.create_run(self.experiment_id, tags={**self.tags, "mlflow.runName": name})
        run_id = run.info.run_id
        stamp = int(time.time() * 1000)
        self.client.log_batch(
            run_id,
            metrics=[Metric(k, float(v), stamp, 0) for k, v in metrics.items()],
            params=[
                Param(k, json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v))
                for k, v in parameters.items()
            ],
        )
        for path in artifacts or []:
            self.client.log_artifact(run_id, str(path))
        self.client.set_terminated(run_id)
        return run_id
