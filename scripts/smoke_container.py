"""Verify an image fails closed without the ignored model. Safe for public CI."""

import json
import subprocess
import time
from pathlib import Path

import requests


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, stderr=subprocess.STDOUT).strip()


def main():
    container = docker(
        "run",
        "-d",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges:true",
        "-p",
        "127.0.0.1::8000",
        "readmitiq-api:phase6",
    )
    try:
        address = docker("port", container, "8000/tcp").splitlines()[0]
        for _ in range(60):
            try:
                health = requests.get(f"http://{address}/health", timeout=2)
                break
            except requests.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("Container did not start")
        assert health.status_code == 503 and not health.json()["model_loaded"]
        metadata = requests.get(f"http://{address}/v1/model", timeout=3)
        assert metadata.status_code == 503 and metadata.json()["error"] == "model_missing"
        payload = json.loads(Path("examples/synthetic/single.json").read_text())
        response = requests.post(f"http://{address}/v1/predict", json=payload, timeout=3)
        assert response.status_code == 503 and response.json()["error"] == "model_missing"
        user = docker("exec", container, "id", "-u")
        assert user == "10001"
        logs = docker("logs", container)
        assert '"event": "model_unavailable"' in logs
        report = {
            "image": "readmitiq-api:phase6",
            "user_id": int(user),
            "missing_artifact_health_status": health.status_code,
            "missing_artifact_prediction_status": response.status_code,
            "artifact_verified": False,
            "real_artifact_test": False,
            "result": "passed: image fails closed without the frozen artifact",
        }
        Path(".cache").mkdir(exist_ok=True)
        Path(".cache/phase6-container-contract.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))
    finally:
        docker("rm", "-f", container)


if __name__ == "__main__":
    main()
