# CI reproduction review

The initial Linux run failed because Python 3.12 delegates zero-timestamp gzip compression to
zlib, which inserts host-specific OS metadata. The original macOS files use byte 19; Linux
uses byte 3. Serialization now fixes the original byte, UTF-8, LF, level 9 and timestamp zero.
All four original compressed-file hashes, assignments, counts, seed, policy and freeze date
are preserved. The complete manifest guard remains strict. Candidate files are staged and
are installed only after verification; failed reconstruction cannot leave a replacement lock.

CI runs inspection/EDA before freezing, then `make split`, then `make phase3`. The notebook
target also depends on `split`. Freezing before EDA would violate the existing test-access
guard. The allocator already sorts patients and uses seeded stratification; it was not changed.

## Separate model-report portability review

After the split passed, the report's reviewed-value check exposed a small random-forest
difference between the original macOS arm64 run and Linux x86_64 / Python 3.12.14. This is
an observed environment difference; its numerical cause inside the estimator has not been
isolated. Identical serialized data does not establish bit-identical fitted estimators.

The complete aggregate metrics and paired comparisons were reviewed from
[Linux diagnostic run 35794732316](https://github.com/Ajay0612/readmit-iq/actions/runs/35794732316).
This run intentionally retained the old report guard and failed before publishing new prose.

| Random-forest metric | Original macOS | Linux reproduction |
| --- | ---: | ---: |
| Average precision | 0.197471 | 0.197713 |
| ROC-AUC | 0.658244 | 0.658434 |
| Brier score | 0.094665 | 0.094652 |
| AP difference vs logistic | 0.003188 | 0.003430 |
| Paired 95% lower bound | -0.003366 | -0.003170 |
| Paired 95% upper bound | 0.010637 | 0.011526 |
| TN / FP / FN / TP at 0.50 | 12091 / 0 / 1499 / 0 | 12091 / 0 / 1499 / 0 |

Other models' AP, ROC-AUC and confusion counts agree at eight decimal places; negligible
floating-point Brier differences do not change the conclusions. Boosting remains ahead of
forest and logistic. Forest's paired AP interval still spans zero. Prior-utilization,
diagnosis, class-weighting and error-analysis conclusions are unchanged. Test was not evaluated.

The report recognizes these two explicitly reviewed forest AP values using the existing
absolute tolerance of 0.00001. It does not accept an unrestricted range of results. Additional
checks reject a changed ranking, inconsistent paired estimate, interval excluding zero or
changed forest confusion counts. All other reviewed AP and high-utilization error checks stay
in force. Checks run before any report/figure writes. Forest prose renders the current paired
estimate and interval instead of copying the macOS numbers into a Linux report.

The committed Phase 3 model outputs remain the original macOS run. No source data, split
manifest, model settings, feature policy, thresholds or Phase 4 recommendations were changed.
Only aggregate validation CSVs are printed in failure diagnostics; datasets, per-encounter
predictions, credentials and fitted models are not uploaded.
