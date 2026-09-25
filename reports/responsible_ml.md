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

## Test diagnostics, frozen primary

| group | level | encounters | patients | positives | suppressed | prevalence | average_precision | recall | recall_ci_low | recall_ci_high | precision | false_negative_rate | mean_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prior_inpatient | Any | 4468 | 2852 | 739 | False | 0.1654 | 0.2608 | 0.3965 | 0.3507 | 0.4423 | 0.2552 | 0.6035 | 0.1533 |
| prior_inpatient | None | 9081 | 8569 | 748 | False | 0.0824 | 0.1347 | 0.0762 | 0.0572 | 0.0952 | 0.2767 | 0.9238 | 0.0882 |
| prior_inpatient_high | 0–2 | 12679 | 9631 | 1263 | False | 0.0996 | 0.1649 | 0.1219 | 0.1039 | 0.1400 | 0.2508 | 0.8781 | 0.0992 |
| prior_inpatient_high | 3+ | 870 | 485 | 224 | False | 0.2575 | 0.3643 | 0.8750 | 0.8303 | 0.9197 | 0.2649 | 0.1250 | 0.2624 |
| age | [0-10) | 24 | 22 | 2 | True | — | — | — | — | — | — | — | — |
| age | [10-20) | 68 | 59 | 5 | True | — | — | — | — | — | — | — | — |
| age | [20-30) | 214 | 162 | 25 | True | — | — | — | — | — | — | — | — |
| age | [30-40) | 525 | 397 | 52 | False | 0.0990 | 0.2785 | 0.3269 | 0.1032 | 0.5507 | 0.3269 | 0.6731 | 0.1146 |
| age | [40-50) | 1327 | 988 | 119 | False | 0.0897 | 0.1880 | 0.1849 | 0.1092 | 0.2605 | 0.2418 | 0.8151 | 0.0966 |
| age | [50-60) | 2361 | 1750 | 245 | False | 0.1038 | 0.2860 | 0.2898 | 0.1973 | 0.3823 | 0.3880 | 0.7102 | 0.0975 |
| age | [60-70) | 3032 | 2239 | 337 | False | 0.1111 | 0.1855 | 0.2047 | 0.1578 | 0.2517 | 0.2447 | 0.7953 | 0.1082 |
| age | [70-80) | 3441 | 2484 | 417 | False | 0.1212 | 0.2249 | 0.2686 | 0.2120 | 0.3252 | 0.2494 | 0.7314 | 0.1189 |
| age | [80-90) | 2216 | 1619 | 238 | False | 0.1074 | 0.1594 | 0.2059 | 0.1506 | 0.2611 | 0.1922 | 0.7941 | 0.1197 |
| age | [90-100) | 341 | 252 | 47 | False | 0.1378 | 0.1841 | 0.1702 | 0.0741 | 0.2663 | 0.2581 | 0.8298 | 0.1129 |
| gender | Female | 7277 | 5190 | 797 | False | 0.1095 | 0.2012 | 0.2359 | 0.1978 | 0.2740 | 0.2590 | 0.7641 | 0.1091 |
| gender | Male | 6272 | 4567 | 690 | False | 0.1100 | 0.2187 | 0.2348 | 0.1899 | 0.2797 | 0.2580 | 0.7652 | 0.1103 |
| race | AfricanAmerican | 2576 | 1745 | 294 | False | 0.1141 | 0.2127 | 0.2347 | 0.1532 | 0.3162 | 0.2323 | 0.7653 | 0.1125 |
| race | Asian | 66 | 59 | 1 | True | — | — | — | — | — | — | — | — |
| race | Caucasian | 10155 | 7339 | 1142 | False | 0.1125 | 0.2116 | 0.2391 | 0.2079 | 0.2702 | 0.2698 | 0.7609 | 0.1101 |
| race | Hispanic | 236 | 196 | 14 | True | — | — | — | — | — | — | — | — |
| race | Other | 211 | 166 | 20 | True | — | — | — | — | — | — | — | — |
| race | Unknown | 305 | 278 | 16 | True | — | — | — | — | — | — | — | — |

Material mean probability-bias flags: **0** of 37
supported group rows. None meet the prespecified joint size/magnitude/interval criterion.
The [complete table](modeling/phase5/test/subgroups.csv) includes Brier, bias intervals,
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
