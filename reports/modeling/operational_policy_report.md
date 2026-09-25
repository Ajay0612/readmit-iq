# Validation-selected operational policy

Prioritize the **top 10% highest-risk eligible discharge encounters**, scored at confirmed
discharge after destination is known. This is a **portfolio capacity scenario assumption**,
selected without actual staffing, intervention costs or clinical utility estimates. The user
accepted using a recommended illustrative scenario. It is not a proposed clinical standard.

Top 5% concentrates yield but captures only 13.3% of readmissions. Increasing to 10% captures
21.4%, with precision 23.6%; 15% captures 28.4% at 20.9% precision. Ten percent illustrates
a constrained, understandable staffing budget. No cost-optimal policy is claimed, and F1
was not optimized. Larger capacities below remain sensitivity summaries, not alternate
post-test candidates.

| capacity | cutoff | targeted_encounters | targeted_patients | tp | fn | recall | precision | lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0500 | 0.2220 | 679 | 440 | 199 | 1300 | 0.1328 | 0.2931 | 2.6571 |
| 0.1000 | 0.1694 | 1359 | 871 | 321 | 1178 | 0.2141 | 0.2362 | 2.1414 |
| 0.1500 | 0.1431 | 2038 | 1281 | 425 | 1074 | 0.2835 | 0.2085 | 1.8906 |
| 0.2000 | 0.1308 | 2718 | 1732 | 538 | 961 | 0.3589 | 0.1979 | 1.7945 |
| 0.2500 | 0.1218 | 3397 | 2176 | 641 | 858 | 0.4276 | 0.1887 | 1.7107 |

Rank scores descending, tie-break by SHA-256 of `42:<encounter_id>` ascending, then select
`floor(0.10*N)`. IDs only break ties; outcomes never allocate capacity. Validation cutoff is
0.169407398, but that probability is **not** the deployed rule. Each unlabeled batch's score
distribution determines its cutoff. The same fraction/tie policy will apply to frozen test.

**Prediction/outreach unit: encounter.** Validation targets 1,359 discharge encounters from
871 unique historical people. Patient counts deduplicate the observed partition; they cannot
be summed across capacities/deciles or scaled to 10,000 discharges. No encounter dates are
available to estimate simultaneous caseload, recurring contacts or daily/weekly batches.
Real scheduling, repeat-contact handling and the batch boundary require prospective workflow
design. Pooled historical ranking does not validate a streaming allocation system.

## Fixed probability thresholds are comparison points

| threshold | targeted_encounters | targeted_patients | fraction_targeted | recall | precision | specificity | fp | fn | number_needed_to_contact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0500 | 13377 | 9577 | 0.9843 | 0.9953 | 0.1115 | 0.0170 | 11885 | 7 | 8.9658 |
| 0.1000 | 5574 | 3608 | 0.4102 | 0.6064 | 0.1631 | 0.6142 | 4665 | 590 | 6.1320 |
| 0.1500 | 1823 | 1144 | 0.1341 | 0.2675 | 0.2200 | 0.8824 | 1422 | 1098 | 4.5461 |
| 0.1694 | 1359 | 871 | 0.1000 | 0.2141 | 0.2362 | 0.9142 | 1038 | 1178 | 4.2336 |
| 0.2000 | 906 | 588 | 0.0667 | 0.1621 | 0.2682 | 0.9452 | 663 | 1256 | 3.7284 |
| 0.2500 | 482 | 298 | 0.0355 | 0.1001 | 0.3112 | 0.9725 | 332 | 1349 | 3.2133 |
| 0.3000 | 253 | 142 | 0.0186 | 0.0567 | 0.3360 | 0.9861 | 168 | 1414 | 2.9765 |
| 0.5000 | 31 | 14 | 0.0023 | 0.0067 | 0.3226 | 0.9983 | 21 | 1489 | 3.1000 |

The listed 0.169407 comparison is rounded; the rank policy enforces capacity and handles ties
explicitly. A 0.50 threshold misses almost every event and has no useful capacity rationale.

## Illustrative operational scenario per 10,000 eligible discharges

Scale validation's encounter-level experience: **1,103 expected recorded readmissions**,
**1,000 outreach encounters**, **236 readmissions surfaced** by model targeting versus **110**
under random outreach, or **126 additional surfaced readmissions**. About **4.23 contacts per
observed readmission surfaced** is a retrospective yield measure, not number needed to treat.
These are expected scaled counts, not a forecast for a named hospital or unique-person counts.
No effect on readmission, prevention, cost saving or intervention effectiveness is estimated.

## Proposed use pending external validation

A care-management team could review an available discharge batch, prioritize its top 10%,
and verify patient context before offering follow-up. Unflagged patients retain usual care;
the model should not make diagnoses, deny services or determine discharge destinations.
The mixed rehabilitation population and strong destination signal require explicit review.
Actual staffing, field arrival times, outreach acceptance and outcomes must be established
before this retrospective scenario can support a hospital policy.

[Risk deciles](phase5/validation/risk_deciles.csv) include observed/predicted risk, unique
people, cumulative event capture and lift. Decile 1 is the highest-risk tenth, with 321
readmissions and 23.6% observed risk; ten deciles reconcile to all 13,590 encounters/1,499 events.
