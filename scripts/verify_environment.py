"""Exercise Phase 1 imports and record the actual interpreter/package versions."""

import importlib
import json
import os
import platform
import random
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/ipython"))

from readmit_iq.config import load_config  # noqa: E402

config = load_config()
random.seed(config.random_seed)
packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "PyYAML": "yaml",
    "pydantic": "pydantic",
    "requests": "requests",
    "joblib": "joblib",
    "jupyterlab": "jupyterlab",
    "ipykernel": "ipykernel",
    "nbclient": "nbclient",
    "pytest": "pytest",
    "ruff": "ruff",
    "pre-commit": "pre_commit",
    "pip-tools": "piptools",
}
for module in packages.values():
    importlib.import_module(module)
import numpy as np  # noqa: E402

rng = np.random.default_rng(config.random_seed)
assert np.isfinite(rng.normal(size=10)).all()
assert sys.version_info >= (3, 11)
assert sys.prefix != sys.base_prefix, "Use the project's .venv interpreter"
result = {
    "python": platform.python_version(),
    "platform": platform.system(),
    "machine": platform.machine(),
    "in_virtual_environment": True,
    "seed": config.random_seed,
    "imports_passed": True,
    "packages": {name: version(name) for name in packages},
    "scope": "Phase 1 dependencies only; modeling/API/demo extras are deferred",
}
config.report_dir.mkdir(parents=True, exist_ok=True)
(config.report_dir / "environment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
