"""Phase 3 configuration with paths rooted at the existing project contract."""

from pathlib import Path

import yaml

from readmit_iq.config import load_config


def settings() -> tuple[Path, dict]:
    root = load_config().raw_dir.parent.parent
    policy = yaml.safe_load((root / "configs/phase3.yaml").read_text())
    if policy["random_seed"] != 42 or policy["phase"] != 3:
        raise ValueError("Review the frozen Phase 3 seed and policy before changing them")
    return root, policy
