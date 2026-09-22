"""Small report writers shared by the Phase 2 audit and EDA."""

import json
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "&#124;").replace("\n", " ")

    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    rows.extend(
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
