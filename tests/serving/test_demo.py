"""CSV adapter and Streamlit rendering use an explicitly mocked HTTP boundary."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

try:
    version("streamlit")
except PackageNotFoundError:
    pytest.skip("UI tests run in the isolated demo environment", allow_module_level=True)
from streamlit.testing.v1 import AppTest  # noqa: E402

from readmit_iq.serving.demo_client import DemoError, csv_to_batch  # noqa: E402
from readmit_iq.serving.examples import synthetic_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_bundled_csv_is_exactly_the_hand_authored_synthetic_batch():
    content = (ROOT / "examples/synthetic/batch.csv").read_bytes()
    assert csv_to_batch(content) == synthetic_batch()


@pytest.mark.parametrize(
    "fault", ["header", "fraction", "negative", "duplicate_column", "empty", "encoding"]
)
def test_csv_rejects_malformed_values_without_exposing_payload(fault):
    content = (ROOT / "examples/synthetic/batch.csv").read_text()
    if fault == "header":
        content = content.replace("request_id", "PRIVATE")
    if fault == "fraction":
        content = content.replace(",2,0,0,0,", ",2.5,0,0,0,", 1)
    if fault == "negative":
        content = content.replace(",2,0,0,0,", ",-2,0,0,0,", 1)
    if fault == "duplicate_column":
        content = content.replace("number_inpatient", "number_outpatient", 1)
    if fault == "empty":
        content = content.splitlines()[0]
    raw = b"\xff\xfe" if fault == "encoding" else content.encode()
    with pytest.raises(DemoError) as caught:
        csv_to_batch(raw)
    assert "PRIVATE" not in str(caught.value)


def test_streamlit_single_and_batch_call_http_client(monkeypatch):
    calls = []

    def mock_http(method, endpoint, payload=None):
        calls.append((method, endpoint, payload))
        if endpoint == "/v1/predict":
            return dict(
                risk_percent=12.345,
                model_version="contract-test-double",
                decision_context="Synthetic contract test",
            )
        return dict(
            total_eligible_encounters=20,
            number_selected=2,
            model_version="contract-test-double",
            policy_version="test-policy",
            decision_context="Portfolio assumption",
            predictions=[
                dict(
                    request_id=f"demo-{i}",
                    rank=i + 1,
                    risk_percent=20 - i / 2,
                    selected_for_outreach=i < 2,
                )
                for i in range(20)
            ],
        )

    monkeypatch.setattr("readmit_iq.serving.demo_client.api_request", mock_http)
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py")).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception and at.metric[0].value == "12.3%"
    assert any("requires comparison" in item.value for item in at.info)
    at.button[1].click().run()
    assert not at.exception and len(at.dataframe) == 1
    assert [c[1] for c in calls] == ["/v1/predict", "/v1/prioritize"]


def test_streamlit_handles_api_unavailability(monkeypatch):
    def unavailable(*args):
        raise DemoError("The local API is unavailable.")

    monkeypatch.setattr("readmit_iq.serving.demo_client.api_request", unavailable)
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py")).run()
    at.button[0].click().run()
    assert not at.exception and "unavailable" in at.error[0].value
