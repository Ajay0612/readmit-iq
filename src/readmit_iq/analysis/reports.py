"""Readable reports driven by computed tables, with policy decisions stated explicitly."""

from pathlib import Path

import pandas as pd

from readmit_iq.analysis.reporting import markdown_table


def cohort_report(
    path: Path,
    flow: pd.DataFrame,
    disposition: pd.DataFrame,
    sensitivity: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> None:
    text = [
        "# Analytical cohort: discharge outreach",
        "",
        "Predict the source `<30` readmission label for confirmed live discharges to home,",
        (
            "outpatient follow-up, or selected postacute settings. "
            "Scoring occurs after the destination"
        ),
        "is confirmed and the available encounter record is assembled. This is a project-specific",
        "outreach population, not a recreation of a CMS quality measure.",
        "",
        "## Eligibility rationale",
        "",
        "- Death-coded dispositions cannot support live-patient outreach. Hospice is excluded",
        (
            "  because its goals require a separate care pathway, not "
            "because readmission is impossible."
        ),
        "  The hospice rows actually contain 43 `<30` labels; death-coded rows contain zero.",
        "- Explicit hospital/inpatient transfers do not identify the completed episode needed for",
        "  post-discharge outreach. No timestamps allow transfer chains to be joined. Codes 5",
        "  and 27 have broad descriptions; their conservative exclusion is a project assumption.",
        (
            "- Codes 9 and 12 do not confirm discharge; 15 is a "
            "within-institution swing-bed continuation."
        ),
        "  Those cases need a separately specified transition-of-care workflow.",
        "- Unknown, unmapped and insufficiently specified destinations cannot verify eligibility.",
        "  They are excluded from the primary cohort, not recoded as safe home discharges. The",
        "  retained-unknown sensitivity quantifies this assumption's population effect.",
        (
            "- SNF, intermediate-care, nursing-facility and "
            "rehabilitation destinations remain eligible"
        ),
        (
            "  for coordinated postacute follow-up. Code 22 includes "
            "hospital rehab units, so its mixed"
        ),
        "  setting is an explicit uncertainty; a no-rehabilitation sensitivity is included.",
        (
            "- Left-against-medical-advice encounters are retained: the "
            "hospital may still offer outreach."
        ),
        "  No age-based exclusion or first-encounter-per-patient filtering is imposed.",
        "",
        "These decisions were based on category meaning rather than optimizing label prevalence.",
        "No care benefit or avoidability is inferred from the label. All raw bytes are unchanged.",
        "",
        "## Exclusions (mutually exclusive)",
        "",
        markdown_table(exclusions),
        "",
        "## Flow and prevalence",
        "",
        markdown_table(flow),
        "",
        "## Sensitivity to cohort assumptions",
        "",
        markdown_table(sensitivity),
        "",
        "A changed prevalence describes the chosen population; it is not evidence that a model",
        "improves. The mixed rehabilitation setting materially changes the population's rate,",
        "so its inclusion should be settled with the intended workflow before Phase 3.",
        "",
        "## Every source disposition code",
        "",
        markdown_table(disposition),
        "",
        "Descriptions are from the checksum-verified `IDS_mapping.csv`, including unused codes.",
        "These historical source mappings take precedence over modern code lists.",
        "",
        "## Target and reproducibility",
        "",
        "`<30 → 1`, `>30 → 0`, `NO → 0`. Unexpected labels fail before any cohort exclusion.",
        "This approximates the business question using recorded less-than-30-day readmission.",
        "The inclusive day-30 boundary, out-of-network events, planned readmissions and complete",
        "follow-up are unverified. The original label remains alongside `readmitted_lt30`.",
        "Rules are versioned in `configs/phase2.yaml` and applied by `data/cohort.py`.",
        "`data/interim/eligible_encounters.csv` and `cohort_membership.csv` are ignored by Git.",
        "Only fixed `? → missing` normalization and the target were added; no learned transform",
        "was fit. Administrative codes and lab `None` remain intact.",
        "",
        "## Sources and scope",
        "",
        "- [UCI release and variable descriptions](https://doi.org/10.24432/C5230J).",
        "- [Strack et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/): the original",
        "  analysis excluded death/hospice and used one encounter per patient. This cohort retains",
        "  repeated encounters and therefore does not reproduce that study population.",
        (
            "- [CMS/Yale 2011 draft methodology, section "
            "2.3.2](https://www.cms.gov/medicare/quality-initiatives-patient-assessment-instruments/mms/downloads/mmshospital-wideall-conditionreadmissionrate.pdf):"
        ),
        "  supports distinguishing live discharges and acute transfer episodes. Its Medicare-age",
        "  and AMA quality-measure rules are not adopted for this different outreach use case.",
    ]
    path.write_text("\n".join(text) + "\n")


def audit_report(path: Path, audit: pd.DataFrame) -> None:
    lines = [
        "# Feature audit and leakage review",
        "",
        "All 50 original fields are audited below. Detailed source definitions, raw/cohort dtypes,",
        "raw cardinality, observed low-cardinality values, missing markers and policy decisions",
        "are machine-readable in `feature_audit.csv`. Semantic definitions also remain in",
        "[`docs/data_dictionary.md`](../../docs/data_dictionary.md). No predictor matrix is built.",
        "",
        "## Scoring contract",
        "",
        "Score at completed discharge, after destination confirmation. 'Definitely safe' below",
        "means the source definition is antecedent to the outcome; it does not prove live-system",
        "availability because this extract has no field-level timestamps. For retrospective",
        "encounter summaries, finalized diagnoses and billing fields, verify actual arrival time",
        "before deployment. Preserve questionable fields for Phase 3 sensitivity experiments.",
        "",
        "Discharge disposition is not mechanically future leakage at this scoring time. It defines",
        (
            "eligibility and may encode destination/care intensity, but "
            "may be finalized late or change"
        ),
        (
            "under an intervention. Compare models with and without it, "
            "conditional on the same cohort."
        ),
        (
            "Diagnosis counts/codes and medication/procedure summaries "
            "may be finalized after discharge;"
        ),
        "they belong in a timing-sensitive ablation. No post-readmission measurement is knowingly",
        "used. An admission-time use case would require a substantially different feature set.",
        "",
        "## Clear modeling exclusions",
        "",
        (
            "`encounter_id`, `patient_nbr`: identifiers. `readmitted` "
            "and the added `readmitted_lt30`: outcomes."
        ),
        (
            "`examide`, `citoglipton`: constant. `weight`: initially "
            "omit for sparse coverage, retain for"
        ),
        "sensitivity. These are modeling recommendations; columns remain in the analytical file.",
        "Full-dataset encounter counts/repeat status and future patient outcomes are analysis-only",
        "and must never become predictor columns. Do not infer chronology from encounter IDs.",
        "",
        "## Per-variable review",
        "",
        (
            "Unknown percentages include `?`, parser NA, administrative "
            "unknown codes and invalid gender,"
        ),
        "as applicable. Not-measured lab `None` is reported separately and is not missingness.",
    ]
    for group, part in audit.groupby("group", sort=False):
        lines += ["", f"### {group.capitalize()}", ""]
        columns = [
            "variable",
            "semantic_type",
            "cohort_cardinality",
            "cohort_unknown_pct",
            "cohort_not_measured_count",
        ]
        lines += [
            markdown_table(part[columns]),
            "",
            markdown_table(
                part[["variable", "leakage_review", "availability_at_discharge", "treatment"]]
            ),
        ]
    path.write_text("\n".join(lines) + "\n")


def missingness_report(
    path: Path, ranked: pd.DataFrame, comparisons: pd.DataFrame, contrasts: dict
) -> None:
    display = comparisons.copy()
    for column in ["rate", "ci_low", "ci_high"]:
        display[column + "_pct"] = 100 * display[column]
    text = [
        "# Missingness strategy",
        "",
        "Source `?` values are normalized to missing on a copied eligible table. No mean/mode",
        "imputation, rare-level merging, scaling or encoding is fit on the full dataset.",
        "",
        "## Ranked coverage and treatment",
        "",
        markdown_table(ranked),
        "",
        "## Is unknown status associated with the outcome?",
        "",
        markdown_table(
            display[
                [
                    "feature",
                    "level",
                    "encounters",
                    "positives",
                    "rate_pct",
                    "ci_low_pct",
                    "ci_high_pct",
                    "sparse",
                ]
            ]
        ),
        "",
        "Confidence intervals use patient-cluster variance. Small groups have no normal interval.",
        "These descriptive differences do not show why a value is missing; no MCAR/MAR/MNAR claim",
        "or causal interpretation can be established from this extract.",
        "",
        "## Decisions for Phase 3",
        "",
        (
            "- Weight: only 3,013 eligible encounters have recorded "
            "values. Unknown and recorded groups"
        ),
        (
            "  have very similar outcome rates. Omit from the first "
            "model; do not invent weights for the"
        ),
        "  remaining population. A small observed subset cannot establish absence of within-weight",
        "  associations. An explicit coverage ablation can be considered after the first model.",
        "- Medical specialty: keep explicit Unknown and test a with/without-feature ablation.",
        (
            "  Missingness has an observed rate difference, but can "
            "encode hospital documentation habits."
        ),
        "- Payer: preserve Unknown and retain for a timed-availability/access-proxy ablation. Its",
        (
            "  missingness alone shows little difference, which does not "
            "prove the whole feature useless."
        ),
        "- Race: preserve Unknown, never infer or mode-fill race. Missingness relates to observed",
        "  outcomes and warrants subgroup/coverage audits; no fairness claim follows from EDA.",
        "- Diagnoses: retain explicit Unknown at encoding time. Missing primary diagnoses are too",
        (
            "  sparse for dependable inference. Do not replace absent "
            "diagnoses with a normal category."
        ),
        "- Admission codes: map documented unavailable/NULL/not-mapped codes to explicit Unknown",
        "  inside the future pipeline, preserving the raw codes for audit. All primary-cohort",
        "  discharge destinations are known by design.",
        (
            "- Gender Unknown/Invalid: preserve an unknown category and "
            "its three rows; do not infer gender."
        ),
        "- Lab None: retain as Not measured, distinct from an unknown result or a normal test.",
        "- Low-frequency categorical levels: learn any grouping threshold on training data only.",
        "",
        "Supported unknown-minus-observed differences (percentage points):",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "feature": name,
                        "difference_pp": value["difference_pp"],
                        "CI_low_pp": value["difference_ci_low_pp"],
                        "CI_high_pp": value["difference_ci_high_pp"],
                    }
                    for name, value in contrasts.items()
                ]
            )
        ),
    ]
    path.write_text("\n".join(text) + "\n")
