"""Apply the original outcome-independent Phase 5 ranking without changing its policy."""

from readmit_iq.decision_support.ranking import capacity_count, ranked_indices
from readmit_iq.serving.contract import CONTRACT
from readmit_iq.serving.schemas import PrioritizationResponse, RankedPrediction


def prioritize(predictor, request):
    probabilities = predictor.probabilities([item.features for item in request.records])
    fraction = predictor.model.metadata["targeting"]["fraction"]
    # request_id is the batch encounter's stable, non-predictive tie key. No generated
    # row-position key is substituted; shuffling the batch cannot change tied selection.
    order = ranked_indices(probabilities, [r.request_id for r in request.records], seed=42)
    selected = capacity_count(len(order), fraction)
    return PrioritizationResponse(
        model_version=predictor.model.metadata["version"],
        policy_version=CONTRACT["policy_version"],
        total_eligible_encounters=len(order),
        number_selected=selected,
        capacity_fraction=fraction,
        capacity_is_assumed=True,
        decision_context=(
            "Top 10% is a portfolio capacity assumption. "
            "Batches with fewer than ten encounters select zero; standard care is unchanged."
        ),
        predictions=[
            RankedPrediction(
                request_id=request.records[int(index)].request_id,
                readmission_probability=float(probabilities[index]),
                risk_percent=float(probabilities[index] * 100),
                rank=rank,
                selected_for_outreach=rank <= selected,
            )
            for rank, index in enumerate(order, 1)
        ],
    )
