# Synthetic demonstration inputs

These records were hand-authored for API/UI demonstrations. They were not copied or sampled
from the historical dataset. They contain no real patient identifiers or readmission outcomes.

- `single.json`: low prior-utilization pattern.
- `prior_utilization.json`: prior inpatient/emergency/outpatient utilization pattern.
- `discharge_context.json`: rehabilitation destination context associated with higher model risk.
- `batch.json` / `batch.csv`: the same 20 synthetic records, repeating three patterns to exercise
  deterministic ties. The frozen 10% policy selects exactly two.

Patterns illustrate model associations, not clinical advice or causal effects. Stable
`demo-NNNN` IDs are non-predictive tie keys. See [local demo instructions](../../docs/local_demo.md).
