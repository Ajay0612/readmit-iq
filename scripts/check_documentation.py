"""Check portfolio links and claims against committed aggregates; never open encounter data."""

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")
IMAGE_LINK = re.compile(r"\)\]\(([^\s)]+)\)")


def without_fences(text):
    return re.sub(r"^```.*?^```\s*$", "", text, flags=re.MULTILINE | re.DOTALL)


def heading_anchors(text):
    counts, anchors = {}, set()
    for heading in re.findall(r"^#{1,6}\s+(.+)$", without_fences(text), flags=re.MULTILINE):
        heading = re.sub(r"<[^>]*>", "", heading).lower().strip()
        slug = re.sub(r"[^\w\s-]", "", heading).replace(" ", "-")
        repeat = counts.get(slug, 0)
        counts[slug] = repeat + 1
        anchors.add(f"{slug}-{repeat}" if repeat else slug)
    return anchors


def main():
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    documents += sorted((ROOT / "reports/portfolio").glob("*.md"))
    checked, external, errors = 0, set(), []
    for document in documents:
        text = without_fences(document.read_text())
        for match in [*LINK.finditer(text), *IMAGE_LINK.finditer(text)]:
            target = match.group(1).strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                external.add(target)
                continue
            path = (document.parent / unquote(url.path)).resolve() if url.path else document
            if not path.exists():
                errors.append(f"{document.relative_to(ROOT)}: missing {target}")
            elif url.fragment and path.suffix == ".md":
                if unquote(url.fragment) not in heading_anchors(path.read_text()):
                    errors.append(f"{document.relative_to(ROOT)}: missing anchor {target}")
            checked += 1
    assert not errors, "\n".join(errors)

    metadata = json.loads((ROOT / "reports/modeling/final_model_metadata.json").read_text())
    primary = next(m for m in metadata["test_metrics"] if m["model"] == metadata["primary"])
    split = json.loads((ROOT / "reports/modeling/split_manifest.json").read_text())
    scenario = json.loads(
        (ROOT / "reports/modeling/phase5/test/business_scenario.json").read_text()
    )
    readme = (ROOT / "README.md").read_text()
    expected = {
        f"{sum(p['encounters'] for p in split['partitions']):,}",
        f"{sum(p['patients'] for p in split['partitions']):,}",
        f"{primary['average_precision']:.5f}",
        f"{primary['roc_auc']:.5f}",
        f"{primary['brier']:.6f}",
        f"{primary['recall']:.2%}",
        f"{primary['precision']:.2%}",
        f"{primary['lift']:.2f}×",
        f"{scenario['model_surfaced_readmissions']:.0f}",
        f"{scenario['random_surfaced_readmissions']:.0f}",
    }
    assert all(value in readme for value in expected), "README headline metric mismatch"
    with (ROOT / "reports/modeling/phase5/test/subgroups.csv").open() as stream:
        prior_use = [
            row
            for row in csv.DictReader(stream)
            if row["model"] == metadata["primary"] and row["group"] == "prior_inpatient"
        ]
    assert len(prior_use) == 2
    assert all(f"{float(row['recall']):.1%}" in readme for row in prior_use)
    assert all(value == 0 for value in split["patient_overlap"].values())
    assert scenario["prevention_effect_estimated"] is False
    figures = set(re.findall(r"!\[[^\]]*\]\((reports/figures/[^)]+)\)", readme))
    assert 5 <= len(figures) <= 7
    figure_record = json.loads((ROOT / "reports/modeling/phase5/verification.json").read_text())
    for name in figures:
        expected_hash = figure_record["figure_sha256"][Path(name).name]
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash
    words = len((ROOT / "docs/project_case_study.md").read_text().split())
    assert 1000 <= words <= 1500, f"Case study length: {words}"
    portfolio = (ROOT / "docs/portfolio_description.md").read_text()
    website = portfolio.split("## Portfolio website description\n")[1].split("## LinkedIn")[0]
    assert 100 <= len(website.split()) <= 150
    notebooks = {}
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert cells and all(c.get("execution_count") is not None for c in cells)
        assert not any(o["output_type"] == "error" for c in cells for o in c.get("outputs", []))
        notebooks[path.name] = len(cells)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    assert not [
        p for p in tracked if p.startswith(("models/", "data/")) and not p.endswith(".gitkeep")
    ]
    assert not [p for p in tracked if p.startswith((".venv/", ".venv-demo/", ".cache/"))]
    media = list((ROOT / "docs/media").glob("*"))
    assert media and sum(p.stat().st_size for p in media if p.is_file()) < 5_000_000
    report = {
        "local_links_checked": checked,
        "external_links_listed_not_network_checked": len(external),
        "headline_metrics_match_committed_aggregates": True,
        "selected_figures_unchanged": sorted(figures),
        "case_study_words": words,
        "portfolio_website_words": len(website.split()),
        "executed_notebook_code_cells": notebooks,
        "tracked_raw_data_or_model_files": 0,
        "demo_media_bytes": sum(p.stat().st_size for p in media if p.is_file()),
        "final_test_records_read": False,
        "model_predictions_invoked": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
