"""FastAPI application with one verified model per process and explicit batch decisions."""

import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from readmit_iq.serving.contract import DISCLAIMER, FEATURES, MAX_BODY_BYTES
from readmit_iq.serving.eligibility import rules
from readmit_iq.serving.errors import ModelIntegrityError, ServingError
from readmit_iq.serving.examples import LOW_HISTORY, synthetic_batch
from readmit_iq.serving.logging import configure_logging
from readmit_iq.serving.model_loader import load_frozen_model
from readmit_iq.serving.predictor import Predictor
from readmit_iq.serving.prioritization import prioritize
from readmit_iq.serving.schemas import (
    BatchRequest,
    BatchResponse,
    EncounterFeatures,
    ErrorResponse,
    HealthResponse,
    ModelResponse,
    PredictionResponse,
    PrioritizationResponse,
)

logger = configure_logging()


def error_response(code, message, status, details=None):
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            error=code,
            message=message,
            details=details or [],
        ).model_dump(),
    )


class BodyLimitMiddleware:
    """Bound bytes before JSON parsing, including chunked/incorrect Content-Length requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        started = False

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.bounded(scope, receive, tracked_send)
        except Exception:
            # Prevent server-level exception logging from leaking an arbitrary exception's
            # message/traceback. All routes return complete, non-streaming JSON responses.
            logger.error(
                "unexpected_error", extra={"error_code": "internal_error", "status_code": 500}
            )
            if scope["type"] == "http" and not started:
                await error_response(
                    "internal_error", "Unexpected server error; no prediction was returned.", 500
                )(scope, receive, send)

    async def bounded(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        data = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(data) + len(chunk) > MAX_BODY_BYTES:
                logger.warning(
                    "request_rejected", extra={"error_code": "body_too_large", "status_code": 413}
                )
                return await error_response(
                    "body_too_large", "Request exceeds the 1 MiB body limit.", 413
                )(scope, receive, send)
            data.extend(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(data), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def create_app(model_loader=load_frozen_model, cohort_rules=None):
    """Dependency injection is for contract tests; deployed app has no fixture/skip-hash mode."""

    @asynccontextmanager
    async def lifespan(application):
        logger.info("service_startup")
        application.state.predictor = None
        application.state.load_error = None
        try:
            model = model_loader()
            application.state.predictor = Predictor(model, cohort_rules or rules())
            logger.info("service_ready", extra={"model_version": model.metadata["version"]})
        except ModelIntegrityError as error:
            application.state.load_error = error
            logger.error("model_unavailable", extra={"error_code": error.code})
        except Exception:
            application.state.load_error = ModelIntegrityError(
                "startup_failed", "Service initialization failed."
            )
            logger.error("model_unavailable", extra={"error_code": "startup_failed"})
        yield
        application.state.predictor = None
        logger.info("service_shutdown")

    application = FastAPI(
        title="ReadmitIQ · Frozen-model portfolio demonstration",
        version="1.0.0",
        description=DISCLAIMER
        + " Single predictions are risk estimates; outreach decisions require a batch.",
        debug=False,
        lifespan=lifespan,
        responses={
            422: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    application.add_middleware(BodyLimitMiddleware)

    def engine():
        predictor = application.state.predictor
        if predictor is None:
            raise application.state.load_error or ServingError(
                "model_unavailable", "Model is unavailable."
            )
        return predictor

    @application.exception_handler(ServingError)
    async def domain_error(request, error):
        logger.warning(
            "request_rejected", extra={"error_code": error.code, "status_code": error.status}
        )
        return error_response(error.code, error.message, error.status, error.details)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        issues = error.errors()
        safe_names = {*FEATURES, "body", "records", "features", "request_id"}
        details = [
            dict(
                location=[
                    part if isinstance(part, int) or part in safe_names else "unknown_field"
                    for part in issue["loc"]
                ],
                code=issue["type"],
                message=issue["msg"],
            )
            for issue in issues[:20]
        ]
        logger.warning("validation_failed", extra={"error_count": len(issues), "status_code": 422})
        return error_response(
            "invalid_input", "Input does not match the v1 encounter contract.", 422, details
        )

    @application.exception_handler(Exception)
    async def unexpected_error(request, error):
        logger.error("unexpected_error", extra={"error_code": "internal_error", "status_code": 500})
        return error_response(
            "internal_error", "Unexpected server error; no prediction was returned.", 500
        )

    def measured(endpoint, count, callback):
        start = time.perf_counter()
        try:
            return callback()
        finally:
            logger.info(
                "inference_request",
                extra={
                    "endpoint": endpoint,
                    "record_count": count,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 3),
                },
            )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["Service"],
        responses={503: {"model": HealthResponse}},
    )
    def health():
        """Readiness is healthy only after frozen artifact integrity and compatibility succeed."""
        predictor = application.state.predictor
        if predictor is None:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="unavailable",
                    model_loaded=False,
                    model_version=None,
                ).model_dump(),
            )
        return HealthResponse(
            status="ok", model_loaded=True, model_version=predictor.model.metadata["version"]
        )

    @application.get("/v1/model", response_model=ModelResponse, tags=["Service"])
    def metadata():
        """Safe provenance and policy metadata; no internal filesystem paths."""
        return engine().public_metadata()

    @application.post("/v1/predict", response_model=PredictionResponse, tags=["Risk estimates"])
    def predict(features: Annotated[EncounterFeatures, Body(examples=[LOW_HISTORY])]):
        """Estimate one eligible discharge's risk. outreach_selected is always null."""
        return measured("predict", 1, lambda: engine().single(features))

    @application.post("/v1/predict/batch", response_model=BatchResponse, tags=["Risk estimates"])
    def predict_batch(batch: Annotated[BatchRequest, Body(examples=[synthetic_batch(10)])]):
        """Score 1–1,000 encounters; preserve input order. No outreach decisions are made."""
        return measured("predict_batch", len(batch.records), lambda: engine().batch(batch))

    @application.post(
        "/v1/prioritize", response_model=PrioritizationResponse, tags=["Batch prioritization"]
    )
    def prioritize_batch(batch: Annotated[BatchRequest, Body(examples=[synthetic_batch(10)])]):
        """Return risk-ranked records and select floor(0.10*N), with the original SHA-256 tie-break.

        Stable unique request_id values serve as non-predictive encounter tie keys. Batches
        below ten select zero. Reject the entire batch if any encounter is ineligible.
        Capacity is a portfolio assumption; clinical deployment is not validated.
        """
        return measured("prioritize", len(batch.records), lambda: prioritize(engine(), batch))

    return application


app = create_app()
