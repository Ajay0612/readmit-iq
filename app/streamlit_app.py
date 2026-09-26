"""Local portfolio UI; Streamlit calls FastAPI and never deserializes the model."""

import pandas as pd
import streamlit as st

from readmit_iq.serving.contract import CONTRACT, DISCLAIMER, read_bundle_json
from readmit_iq.serving.demo_client import DemoError, api_request, csv_to_batch
from readmit_iq.serving.examples import PATTERNS, synthetic_batch

st.set_page_config(page_title="ReadmitIQ · Portfolio Demo", page_icon="◈", layout="wide")
st.title("ReadmitIQ")
st.subheader("30-Day Readmission Risk — Portfolio Demonstration")
st.caption(DISCLAIMER)

with st.sidebar:
    st.markdown("### Frozen model")
    st.write("Logistic regression · unchanged training weights")
    st.caption(CONTRACT["model_version"])
    st.markdown("### How to explore")
    st.write(
        "1. Estimate a synthetic discharge's risk.\n2. Compare a batch at the assumed 10% capacity."
    )
    st.markdown("[Model card](https://github.com/Ajay0612/readmit-iq/blob/main/docs/model_card.md)")
    st.markdown(
        "[Explainability report](https://github.com/Ajay0612/readmit-iq/blob/main/reports/modeling/explainability_report.md)"
    )
    st.caption(
        "Explanations describe associations, not causes. Detailed SHAP is in the linked report."
    )

single, batch = st.tabs(["Single encounter", "Batch prioritization"])
mapping = read_bundle_json("reports/data_quality/id_mapping.json")

with single:
    st.markdown("#### Synthetic demonstration inputs")
    selected_pattern = st.selectbox("Starting example", list(PATTERNS))
    defaults = PATTERNS[selected_pattern]
    with st.form("single_prediction"):
        left, right = st.columns(2)
        features = {}
        with left:
            features["time_in_hospital"] = st.number_input(
                "Time in hospital (days)",
                min_value=1,
                max_value=14,
                value=defaults["time_in_hospital"],
            )
            features["number_inpatient"] = st.number_input(
                "Prior inpatient visits (preceding year)",
                min_value=0,
                max_value=2**31 - 1,
                value=defaults["number_inpatient"],
            )
            features["number_emergency"] = st.number_input(
                "Prior emergency visits (preceding year)",
                min_value=0,
                max_value=2**31 - 1,
                value=defaults["number_emergency"],
            )
            features["number_outpatient"] = st.number_input(
                "Prior outpatient visits (preceding year)",
                min_value=0,
                max_value=2**31 - 1,
                value=defaults["number_outpatient"],
            )
            features["age"] = st.selectbox(
                "Age band",
                CONTRACT["age_bands"],
                index=CONTRACT["age_bands"].index(defaults["age"]),
            )
        with right:
            features["gender"] = st.selectbox(
                "Recorded gender",
                CONTRACT["genders"],
                index=CONTRACT["genders"].index(defaults["gender"]),
            )
            for field, label in [
                ("admission_type_id", "Admission type"),
                ("admission_source_id", "Admission source"),
                ("discharge_disposition_id", "Confirmed discharge destination"),
            ]:
                options = [int(key) for key in mapping[field]]
                features[field] = st.selectbox(
                    label,
                    options,
                    index=options.index(defaults[field]),
                    format_func=lambda code, field=field: f"{code} · {mapping[field][str(code)]}",
                )
            choices = CONTRACT["medical_specialties"]
            features["medical_specialty"] = st.selectbox(
                "Admitting medical specialty",
                choices,
                index=choices.index(defaults["medical_specialty"]),
            )
        submitted = st.form_submit_button("Estimate risk", type="primary")
    if submitted:
        try:
            result = api_request("POST", "/v1/predict", features)
            st.metric("Estimated 30-day readmission risk", f"{result['risk_percent']:.1f}%")
            st.info("Outreach prioritization requires comparison within an operational batch.")
            st.caption(f"Model {result['model_version']} · {result['decision_context']}")
        except DemoError as error:
            st.error(str(error))
    st.caption(
        "Prior inpatient use and destination are strong drivers. "
        "Low prior use can still precede readmission."
    )

with batch:
    st.markdown("#### Compare an available discharge batch")
    st.write(
        "The top 10% is an illustrative outreach capacity. Fewer than ten encounters select zero."
    )
    mode = st.radio(
        "Batch source",
        ["Bundled synthetic batch (20 encounters)", "Upload synthetic CSV"],
        horizontal=True,
    )
    uploaded = None
    if mode == "Upload synthetic CSV":
        st.caption(
            "UTF-8 CSV: request_id + ten features. Maximum 1,000 rows / 1 MiB. Synthetic data only."
        )
        uploaded = st.file_uploader("Synthetic encounter CSV", type=["csv"])
    else:
        st.caption("Twenty synthetic encounters; repeated patterns demonstrate deterministic ties.")
    if st.button("Prioritize batch", type="primary"):
        try:
            if mode == "Upload synthetic CSV":
                if uploaded is None:
                    raise DemoError("Choose a synthetic CSV first.")
                payload = csv_to_batch(uploaded.getvalue())
            else:
                payload = synthetic_batch()
            result = api_request("POST", "/v1/prioritize", payload)
            first, second, third = st.columns(3)
            first.metric("Eligible encounters", result["total_eligible_encounters"])
            second.metric("Prioritized", result["number_selected"])
            third.metric("Assumed capacity", "10%")
            table = pd.DataFrame(result["predictions"])
            table["risk_percent"] = table.risk_percent.round(1)
            st.dataframe(
                table[["rank", "request_id", "risk_percent", "selected_for_outreach"]],
                hide_index=True,
                width="stretch",
            )
            st.bar_chart(
                table.set_index("rank")[["risk_percent"]],
                x_label="Risk rank",
                y_label="Estimated risk (%)",
                color="#287C8E",
            )
            st.caption(result["decision_context"])
            st.caption(f"{result['model_version']} · policy {result['policy_version']}")
        except DemoError as error:
            st.error(str(error))
