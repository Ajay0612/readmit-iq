# Pinned official multi-platform Python 3.12 base; refresh only with compatibility checks.
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS api
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    READMITIQ_SERVING_ROOT=/opt/readmit \
    READMITIQ_MODEL_PATH=/model/readmit_iq_logistic.joblib \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /opt/readmit
COPY requirements-serving.txt ./
RUN python -m pip install --no-cache-dir -r requirements-serving.txt
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/phase2.yaml configs/phase2.yaml
COPY reports/data_quality/id_mapping.json reports/data_quality/id_mapping.json
COPY reports/modeling/final_model_specification.json reports/modeling/final_model_specification.json
COPY reports/modeling/final_model_metadata.json reports/modeling/final_model_metadata.json
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation . \
    && python -m pip check \
    && useradd --create-home --uid 10001 readmit \
    && mkdir /model
USER readmit
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["python", "-m", "uvicorn", "readmit_iq.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM api AS demo
USER root
COPY requirements-demo.txt ./
RUN python -m pip install --no-cache-dir -r requirements-demo.txt && python -m pip check
COPY app/ app/
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USER readmit
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.maxUploadSize=1", "--server.fileWatcherType=none"]

# Plain `docker build .` retains the API as the default application.
FROM api AS default
