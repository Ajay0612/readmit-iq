"""Thin HTTP client and strict CSV transport adapter; no model is loaded in the UI."""

import csv
import io
import os
import re

import requests

from readmit_iq.serving.contract import FEATURES, MAX_BATCH, MAX_BODY_BYTES


class DemoError(ValueError):
    pass


def api_request(method, endpoint, payload=None):
    url = os.environ.get("READMITIQ_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        response = requests.request(method, url + endpoint, json=payload, timeout=(3, 20))
        result = response.json()
    except (requests.RequestException, ValueError) as error:
        raise DemoError(
            "The local API is unavailable. Start it and check its health endpoint."
        ) from error
    if not response.ok:
        # The API supplies sanitized field locations/codes. Do not echo request values.
        message = result.get("message", "The API rejected this request.")
        details = result.get("details", [])
        if details:
            first = details[0]
            message += " " + ".".join(map(str, first["location"])) + ": " + first["message"]
        raise DemoError(message)
    return result


def csv_to_batch(content: bytes):
    if len(content) > MAX_BODY_BYTES:
        raise DemoError("CSV exceeds the 1 MiB upload limit.")
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), strict=True)
        expected = ["request_id", *FEATURES]
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(expected)
            or set(reader.fieldnames) != set(expected)
        ):
            raise DemoError("CSV needs exactly request_id and the ten documented feature columns.")
        records = []
        numeric = {
            "time_in_hospital",
            "number_inpatient",
            "number_emergency",
            "number_outpatient",
            "admission_type_id",
            "admission_source_id",
            "discharge_disposition_id",
        }
        for index, row in enumerate(reader, 1):
            if index > MAX_BATCH:
                raise DemoError(f"CSV exceeds {MAX_BATCH} records.")
            if None in row or None in row.values():
                raise DemoError(f"CSV row {index} has an incorrect number of fields.")
            features = {name: row[name] for name in FEATURES}
            for name in numeric:
                value = features[name]
                if name in {"admission_type_id", "admission_source_id"} and value in {
                    "",
                    "?",
                    "Unknown",
                }:
                    continue
                if not re.fullmatch(r"0|[1-9][0-9]*", value) or len(value) > 10:
                    raise DemoError(f"CSV row {index}: {name} must contain an unsigned integer.")
                features[name] = int(value)
            records.append(dict(request_id=row["request_id"], features=features))
    except (UnicodeDecodeError, csv.Error) as error:
        raise DemoError("Upload a valid UTF-8 CSV with the documented header.") from error
    if not records:
        raise DemoError("CSV must contain at least one encounter.")
    # FastAPI remains the single authority for schemas, eligibility, inference and ranking.
    return {"records": records}
