# Responsible ML: specific findings and deployment limits

The primary model prioritizes repeated prior use much more strongly than first recorded
inpatient history. At validation's 10% budget, no-history recall is 6.0% (95% interval
4.4–7.7%), compared with 38.1% (33.7–42.4%) for any prior use and 87.6% for 3+ visits.
That gap persists despite reasonable average probability alignment: limited-capacity ranking
and calibration answer different questions. It is a major reason to retain standard care
for unflagged patients and evaluate a complementary clinical pathway prospectively.

Recorded female/male validation recall is 20.7%/22.3%, with overlapping cluster intervals.
African American/Caucasian recall is 16.1%/23.2%; the first interval is wide (10.0–22.3%).
Hispanic recall is 13.9% with only 36 positives and a very wide interval. These observational
differences can reflect prevalence, history capture, case mix, missingness and institutional
practice; they establish neither fairness nor unlawful discrimination. No equalized-odds or
causal fairness claim is supported. Race is evaluation-only; recorded binary gender is an
input whose justification and alternatives require local clinical governance.

## Prespecified uncertainty and suppression

Keep Unknown and raw counts. Suppress group performance below **200 encounters, 100 patients,
30 positives or 30 negatives**. Subgroups overlap; patient totals must not be added.
Ratio intervals use whole-patient cluster influence estimates, conditional on the fitted
model and the selected full-partition outreach list. They do not repeat model selection or
rerank subgroups independently. Unlike the headline bootstrap, they do not model outreach
cutoff uncertainty. AP and Brier are descriptive; no multiple-comparison-adjusted hypothesis
test is claimed. Small-group suppression does not establish safety for excluded groups.

Define mean probability bias as predicted mean minus observed prevalence. Flag it as material
only when absolute bias is >=2 percentage points **and** its cluster 95% interval excludes
zero. This is a declared descriptive screen, not proof of calibration. Mean alignment can
hide errors within risk bins; quantile reliability plots and the archived group tables must
be read together. No demographic threshold or subgroup recalibration is fitted.

## Validation diagnostics, frozen primary

| group | level | encounters | patients | positives | suppressed | prevalence | average_precision | recall | recall_ci_low | recall_ci_high | precision | false_negative_rate | mean_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prior_inpatient | Any | 4456 | 2801 | 720 | False | 0.1616 | 0.2418 | 0.3806 | 0.3375 | 0.4236 | 0.2372 | 0.6194 | 0.1528 |
| prior_inpatient | None | 9134 | 8608 | 779 | False | 0.0853 | 0.1374 | 0.0603 | 0.0436 | 0.0771 | 0.2304 | 0.9397 | 0.0878 |
| prior_inpatient_high | 0–2 | 12704 | 9643 | 1273 | False | 0.1002 | 0.1560 | 0.0966 | 0.0804 | 0.1129 | 0.1974 | 0.9034 | 0.0990 |
| prior_inpatient_high | 3+ | 886 | 472 | 226 | False | 0.2551 | 0.3270 | 0.8761 | 0.8344 | 0.9178 | 0.2690 | 0.1239 | 0.2549 |
| age | [0-10) | 30 | 28 | 0 | True | — | — | — | — | — | — | — | — |
| age | [10-20) | 130 | 88 | 7 | True | — | — | — | — | — | — | — | — |
| age | [20-30) | 227 | 170 | 23 | True | — | — | — | — | — | — | — | — |
| age | [30-40) | 488 | 389 | 44 | False | 0.0902 | 0.2147 | 0.1364 | 0.0100 | 0.2627 | 0.1622 | 0.8636 | 0.1013 |
| age | [40-50) | 1374 | 976 | 165 | False | 0.1201 | 0.2608 | 0.2485 | 0.1371 | 0.3599 | 0.2950 | 0.7515 | 0.1076 |
| age | [50-60) | 2334 | 1750 | 223 | False | 0.0955 | 0.2115 | 0.2018 | 0.1407 | 0.2629 | 0.2866 | 0.7982 | 0.0923 |
| age | [60-70) | 2977 | 2190 | 336 | False | 0.1129 | 0.2254 | 0.2530 | 0.1969 | 0.3091 | 0.2787 | 0.7470 | 0.1097 |
| age | [70-80) | 3412 | 2506 | 379 | False | 0.1111 | 0.1758 | 0.2111 | 0.1663 | 0.2558 | 0.2128 | 0.7889 | 0.1152 |
| age | [80-90) | 2213 | 1580 | 275 | False | 0.1243 | 0.1695 | 0.1927 | 0.1413 | 0.2442 | 0.1893 | 0.8073 | 0.1222 |
| age | [90-100) | 405 | 288 | 47 | False | 0.1160 | 0.1330 | 0.0638 | 0.0000 | 0.1326 | 0.0811 | 0.9362 | 0.1112 |
| gender | Female | 7376 | 5232 | 816 | False | 0.1106 | 0.1901 | 0.2071 | 0.1703 | 0.2439 | 0.2230 | 0.7929 | 0.1091 |
| gender | Male | 6214 | 4525 | 683 | False | 0.1099 | 0.2065 | 0.2225 | 0.1839 | 0.2612 | 0.2529 | 0.7775 | 0.1092 |
| race | AfricanAmerican | 2599 | 1811 | 254 | False | 0.0977 | 0.1468 | 0.1614 | 0.0995 | 0.2233 | 0.1614 | 0.8386 | 0.1065 |
| race | Asian | 100 | 76 | 8 | True | — | — | — | — | — | — | — | — |
| race | Caucasian | 10062 | 7227 | 1154 | False | 0.1147 | 0.2114 | 0.2322 | 0.2013 | 0.2632 | 0.2565 | 0.7678 | 0.1107 |
| race | Hispanic | 320 | 236 | 36 | False | 0.1125 | 0.2301 | 0.1389 | 0.0063 | 0.2715 | 0.2273 | 0.8611 | 0.1041 |
| race | Other | 193 | 161 | 20 | True | — | — | — | — | — | — | — | — |
| race | Unknown | 316 | 275 | 27 | True | — | — | — | — | — | — | — | — |

Material mean probability-bias flags: **0** of 39
supported group rows. None meet the prespecified joint size/magnitude/interval criterion.
The [complete table](modeling/phase5/validation/subgroups.csv) includes Brier, bias intervals,
all predeclared clinical/missingness groups and the challenger's identical diagnostics.

## Why this cannot yet guide clinical deployment

* The selected diabetes inpatient cohort dates to **1999–2008**. Coding, clinical treatment,
  discharge practice, population mix and outreach availability may have changed.
* This is observational data with incomplete outside-hospital follow-up; NO is no recorded
  readmission, not verified absence. The exact day-30 boundary cannot be audited from categories.
  No hospital IDs/dates support contemporary site/temporal transport assessment.
* Patient-group splitting prevents repeated-patient overlap (zero across partitions), but
  it is neither temporal nor external validation. Earlier full-cohort EDA exposed eventual
  test outcomes indirectly. Final test is held out from model development, not a fully
  untouched confirmatory study; bootstrap intervals omit that exploration/selection uncertainty.
* Demographics are historical recorded categories with missing/unknown values and small groups.
  Utilization can measure access and fragmented capture as well as illness. Destination and
  specialty can reflect institutional decisions; strong prediction does not imply a causal target.
* Confirmed discharge and destination availability are explicit assumptions. The extract has
  no field arrival timestamps. Audit real-time registration, look-back counts, final versus
  provisional destination, coding corrections and specialty before any shadow evaluation.
* Aggregate retrospective lift measures surfaced events, not preventability or benefit.
  Intervention effect, contact burden, repeated contacts and possible harms are unmeasured.
* Before use: clinical and patient review, current external/temporal validation, silent
  prospective scoring, workflow/capacity evaluation and an intervention study are needed.
  Monitor feature availability, prevalence, calibration, utilization/demographic error gaps,
  workload and outcomes with governance-defined review criteria. No monitoring service is built here.

All model, feature, calibration, subgroup and targeting definitions remain fixed after the
pretest protocol. Final diagnoses, labs, medications and payer remain excluded. This report
documents limitations; it does not authorize direct clinical use or deny care to low scores.
