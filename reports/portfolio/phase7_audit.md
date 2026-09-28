# Final portfolio quality and provenance audit

Phase 7 changes presentation, navigation and documentation. It does not change the model,
feature definitions, preprocessing, final test results or assumed outreach policy. Historical
phase reports remain labeled evidence from their original stage.

## Material reviewed and retained

The review covered the former README, model card, final evaluation/protocol, operational
scenario, selection/calibration/explainability reports, responsible-ML analysis, EDA/modeling
figures, all six executed notebooks, API/OpenAPI, local Streamlit demonstration, tests,
Docker configuration, GitHub Actions and Git history.

The recruiter-facing README now leads with the problem, implemented solution and retrospective
results, followed by methodology, model-selection judgment, business interpretation,
explanations, prominent limitations, demo, architecture and a task-based quick start.
Detailed historical text was moved to [development history](../../docs/development_history.md),
with current setup guidance clearly separated from older commands. Supporting reports and
notebooks were retained rather than edited for appearance.

## Claims and evidence

| Portfolio claim | Committed evidence |
|---|---|
| 90,702 encounters / 65,044 patients; 63,563 / 13,590 / 13,549 split; zero patient overlap | [Split manifest](../modeling/split_manifest.json) |
| AP 0.20837, ROC-AUC 0.65181, Brier 0.094101 | [Final metadata](../modeling/final_model_metadata.json) |
| 23.54% recall, 25.85% precision and 2.36× lift at assumed 10% capacity | [Final report](../modeling/final_test_report.md) |
| About 258 versus 110 surfaced events per 10,000 discharges | [Scenario JSON](../modeling/phase5/test/business_scenario.json) |
| Boosting AP gain +0.00194; interval −0.00427 to +0.00839 | [Selection report](../modeling/model_selection_report.md) |
| Recall 7.6% without prior inpatient use versus 39.6% with prior use | [Responsible ML](../responsible_ml.md) |
| 296 local tests at Phase 6; exact synthetic probability parity | [Serving verification](../serving/phase6_verification.md) |

The review distinguishes validation from test, conditional intervals from selection uncertainty,
encounters from unique people, retrospective yield from intervention effects, and the assumed
capacity from an actual hospital policy. No prevention, savings, clinical deployment,
statistical-equivalence or fairness-certification claim was added.

## Selected visual evidence

| Existing figure | One message | Context and accessibility |
|---|---|---|
| `phase5/02_cumulative_gains.png` | Top 10% captures 23.5% of test events | Labeled validation/test curves and random reference; caption explains encounter ranking. |
| `phase5/01_precision_recall.png` | Ranking improves yield while most events remain outside the queue | Dots mark capacity; caption identifies saved curves and avoids threshold confusion. |
| `phase5/04_operational_scenario.png` | Same assumed workload surfaces about 149 additional events | Direct numeric labels; prevention and savings explicitly excluded. |
| `phase5/05_global_feature_influence.png` | Prior inpatient use and destination dominate explanations | Validation scope and shared comparison sample identified; associations are not causes. |
| `phase5/07_subgroup_recall.png` | Low-history recall is a substantial failure mode | Patient-cluster intervals, overlapping groups and suppression remain visible. |

All five original image files and analytical outputs remain unchanged. Other EDA/CV charts
were reviewed and remain linked through notebooks/history; the README avoids redundant
panels and a very wide exploratory utilization chart. Takeaway titles, restrained color and
supporting captions follow the requested [Storytelling with Data principles](https://www.storytellingwithdata.com/blog/what-your-audience-really-wants).
Every selected chart has a descriptive alt text, a plain-language caption and a full-size link.

Chrome rendering was reviewed at 1280-pixel desktop and 390-pixel mobile widths. Both Mermaid
diagrams render, all five images load, and neither viewport has page-level horizontal overflow
or JavaScript errors. Dense original chart labels can be opened at full resolution; nearby
captions preserve each takeaway on narrow screens. The 19.64-second recording was played and
its single-score, batch-selection and ranked-chart frames inspected. The recording and two
screenshots total 1,613,921 bytes.

The published GitHub README was also checked at desktop and mobile widths: both diagrams
render and the five figure links resolve. The published walkthrough displays both screenshots.
GitHub offers the WebM as a download, so the walkthrough explicitly explains **View raw** and
local playback rather than promising an embedded GitHub player.

## Demonstration and engineering checks

The Phase 7 serving run passed **91 backend tests**, including the four original-artifact
integration cases, plus **9 UI/CSV tests** in the separate UI environment. The backend's UI
module is intentionally skipped because Streamlit runs separately. This does not relabel the
historical 296-test Phase 6 count as a new full-suite run. Artifact integrity and Ruff checks pass.

The actual Docker API/UI was exercised with the hand-authored single example and uploaded
20-record CSV: **6.5%** single risk, **two selections**, rendered ranks and chart, and no browser
errors. [Screenshots and recording](../../docs/demo_walkthrough.md) use synthetic inputs only.
They are application captures, not generated mockups. The recording is abridged, without
narration; the text walkthrough is its accessible alternative. Localhost is never presented
as a public deployment. The original ignored model remains a read-only API mount.

`make docs-check` verifies local file/anchor links, headline metrics against committed
aggregates, figure hashes, requested document lengths, executed notebook outputs, media size
and exclusion of datasets/model binaries from Git. It opens no encounter-level dataset and
invokes no prediction. CI includes this check while preserving all previous protections.

The final documentation check passed **142 local file/anchor links**, checked all **54 executed
code cells across six notebooks** without rerunning them, and confirmed the **1,274-word case
study** and **113-word website description**. A separate HTTP review covered 23 linked URLs,
including local services: 22 returned 200. The PMC source article blocked the automated HTTP
client with 403; its title and article content were verified through the web reader instead.

## Privacy, provenance and repository presentation

New screenshots, examples and recording contain only synthetic tracking IDs. The existing
Phase 1 raw preview explicitly drops `encounter_id` and `patient_nbr`; explanation examples
are anonymous historical illustrations, accurately labeled rather than described as synthetic.
Aggregate ranges/counts are not republished as patient examples. No raw dataset, model binary,
virtual environment, credential or private workspace file is added to Git.

The frozen artifact SHA-256 remains
`ff82996f1f49008dec373655d53cf95a2ce5d940cb7dbc2bd1e25ea6fe447fa8`.
All protected implementation, protocol, specification, archived publication, notebook and
original-figure files are preserved. Neither `make final-eval`, full-cohort EDA nor new model
experiments were run for this phase. Existing CI retains its separate development reproduction
and archived-final verification behavior.

A separate final comparison against the 285 files tracked at Phase 6 found **282 unchanged**.
The only changed existing files are README, Makefile and the CI workflow; additions are the
portfolio documents/media and a documentation checker. All **20 frozen implementation files**
and the archived final publication contract pass the original provenance verifiers. These
checks read committed aggregate evidence, without loading or scoring final-test records.

GitHub description and ten relevant topics were added. The homepage field remains empty;
no public inference URL was invented. **No repository LICENSE exists.** No license or ownership
assumption was introduced; choosing a code license remains the owner's decision. Dataset
attribution remains separate from repository code licensing.

## CI and remaining limitations

[![Current CI](https://github.com/Ajay0612/readmit-iq/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ajay0612/readmit-iq/actions/workflows/ci.yml?query=branch%3Amain)

Follow the badge for the exact current commit and the data-contract/serving-contract job results.
The final completion report records the verified run. CI has no trusted real model binary:
its four integration tests explicitly skip, while local checks use the original artifact.

Historical population selection, incomplete follow-up, unavailable temporal/site validation,
feature timing, low-history errors, subgroup uncertainty and absent prospective benefit evidence
remain material. No clinical readiness, HIPAA compliance or cloud deployment is claimed.
The repository is intended to demonstrate analytical and engineering judgment to recruiters.
