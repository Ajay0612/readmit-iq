# Preliminary feature-engineering plan

Historical Phase 2 discovery plan. See the [Phase 3 ablation evidence](modeling/baseline_model_report.md)
for what was implemented and what did or did not improve validation performance.

Discovery only. No final engineered predictors, fitted preprocessing or models exist. Evidence
uses the 90,702-encounter primary cohort. EDA display bands are not validated prediction thresholds.

Inpatient and emergency rates rise across the displayed 0/1/2/3+ bins, but **none of the three
utilization measures is strictly increasing across all exact-count groups with at least 100
encounters**. Sparse tails fluctuate; no monotonic constraint is justified from these plots alone.

| Candidate | Executed evidence | Leakage / availability concern | Phase 3 recommendation |
|---|---|---|---|
| Separate utilization counts | Inpatient 0/1/2/3+ rates: 8.34/12.72/17.63/26.20%; emergency: 10.34/14.42/18.42/25.40%; outpatient: 10.58/13.72/13.71/13.12% | Source defines preceding-year visits, unlike future full-dataset encounter counts | Keep all three original counts; high priority |
| Any prior inpatient / emergency visit | Clear zero-versus-positive step with continuing gradients | Safe when derived solely from preceding-year counts | Test supplemental indicators without replacing counts |
| Total prior utilization | Pairwise Spearman correlations only 0.156–0.226; rate shapes differ | Equal weighting may obscure distinct care patterns | Test as an addition, not an assumed improvement or replacement |
| High-utilization indicator / count transform | Long right tails and higher 3+ inpatient/ED rates; outpatient levels off | Cutpoint search can overfit | Test a small prespecified set on training-only grouped validation; no arbitrary clipping |
| Nonlinear length of stay | 1–2 vs 8–14 days: 9.08% vs 13.44%; +4.36 pp (95% patient-cluster interval 3.65–5.06) | Known at completed discharge, unavailable at admission | Test flexible numeric effects; retain original count |
| Medication burden bands | 1–9 / 10–19 / 20–29 / 30+ rates: 8.61/11.08/13.00/12.79% | Full-stay count may finalize late; not a discharge prescription list | Test nonlinear effects after timing review; no forced monotonic constraint |
| Broader diagnosis groups | 708/740/776 observed codes in diag_1/2/3; 407/460/496 appear fewer than 20 times; decimal-prefix removal leaves 675/707/744 categories | Final coding may be late; short integer components and E/V codes require care | Verify authoritative ICD-9-CM mapping, version, coverage and unknown/V/E handling before implementing |
| Missing-category indicators | Specialty unknown rate 11.52% vs 10.67% observed; weight unknown 11.08% vs 11.12% observed | Documentation/site proxies may not transport | Explicit Unknown often already encodes missingness; add indicators only when encoding/validation warrants |
| Discharge destination | Rehabilitation 27.70% vs home 9.30%; care settings and severity differ | Mixed rehab category, mutable destination, finalization time | With/without-disposition ablation on the same cohort; no treatment-effect claim |
| Medication change / insulin categories | Change 11.76% vs no change 10.48%; insulin down/up 13.95/13.11% vs no insulin 9.82% | Confounding by indication and full-stay availability | Keep categorical indicators; defer every-drug expansion and causal interpretation |
| Utilization × burden interaction | Marginal gradients suggest a hypothesis; no interaction benefit was demonstrated | Timing and added complexity | Lower priority after simple baselines, evaluated only on development data |

## Avoid or deprioritize

- Identifiers, original/binary outcomes as predictors, future encounter counts, full-dataset repeat
  status and patient-wide outcome aggregates are prohibited.
- Initially omit weight (3,013 observed eligible values) and the two constant medications. Retain
  analytical columns; do not mean/mode-fill sparse weight.
- Diagnosis strings are not continuous numbers. Decimal-prefix reduction is not a clinical grouping.
  This phase inspected syntax and prefixes only.
- Do not infer race/gender or claim fairness from similar marginal rates. Later models need subgroup
  performance and calibration assessment.

Establish the group split and simple baselines before testing these ideas. Select final features
using development performance, timing feasibility and operational usefulness, not EDA alone.

Diagnosis mapping references: [CDC ICD-CM definitions](https://www.cdc.gov/nchs/hus/sources-definitions/icd-cm.htm)
and [CDC ICD-9-CM archive](https://archive.cdc.gov/www_cdc_gov/nchs/icd/icd9cm.htm).
