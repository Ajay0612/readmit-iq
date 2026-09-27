PYTHON ?= .venv/bin/python
RUFF = .venv/bin/ruff
DEMO_PYTHON = .venv-demo/bin/python

.PHONY: install download inspect notebook verify test lint phase1 eda eda-notebook phase2 split baselines baseline-notebook phase3
.PHONY: optimize optimization-reports optimization-notebook phase4
.PHONY: phase5-development final-eval final-notebook phase5
.PHONY: install-phase6 install-demo api demo verify-model phase6-test phase6 docker-build docker-run docker-stop
install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps --no-build-isolation .
download:
	$(PYTHON) -m readmit_iq.data.download
inspect:
	$(PYTHON) -m readmit_iq.data.inspect
notebook:
	$(PYTHON) scripts/execute_notebook.py
verify:
	$(PYTHON) scripts/verify_environment.py
	$(PYTHON) -m pip check
test:
	$(PYTHON) -m pytest
lint:
	$(RUFF) check .
	$(RUFF) format --check .
phase1: download inspect notebook verify lint test
eda:
	$(PYTHON) -m readmit_iq.analysis.run_phase2
eda-notebook:
	$(PYTHON) scripts/execute_notebook.py notebooks/02_eda.ipynb
phase2: eda-notebook verify lint test
split:
	$(PYTHON) -m readmit_iq.modeling.splitting
baselines:
	$(PYTHON) -m readmit_iq.modeling.train
	$(PYTHON) -m readmit_iq.modeling.reports
baseline-notebook: split
	$(PYTHON) scripts/execute_notebook.py notebooks/03_feature_engineering_and_baselines.ipynb
phase3: baseline-notebook verify lint test
optimize:
	$(PYTHON) -m readmit_iq.optimization.fit
optimization-reports:
	$(PYTHON) -m readmit_iq.optimization.evaluate
	$(PYTHON) -m readmit_iq.optimization.reports
optimization-notebook:
	$(PYTHON) scripts/execute_notebook.py notebooks/04_model_optimization.ipynb --timeout 3600
phase4: optimization-notebook verify lint
	$(PYTHON) -m pytest --ignore=tests/serving --junitxml=.cache/phase4-tests.xml
	$(PYTHON) scripts/verify_phase4.py
phase5-development:
	$(PYTHON) scripts/execute_notebook.py notebooks/05_explainability_business_impact.ipynb --timeout 1800
# Explicit one-time command; intentionally never a dependency of phase5 or CI.
final-eval:
	$(PYTHON) -m readmit_iq.decision_support.final_evaluation --execute-frozen-test
final-notebook:
	$(PYTHON) scripts/execute_notebook.py notebooks/06_final_test_evaluation.ipynb --timeout 300
phase5: phase5-development final-notebook verify lint
	$(PYTHON) -m pytest --ignore=tests/serving --junitxml=.cache/phase5-tests.xml
	$(PYTHON) scripts/verify_phase5.py

# Phase 4/5 retain their no-skip historical test evidence. Serving tests have their
# own target/job because their real-artifact and UI dependencies can be absent.
# Phase 6 never trains a model or scores the held-out test set.
install-phase6:
	$(PYTHON) -m pip install -r requirements-phase6.txt
	$(PYTHON) -m pip install --no-deps --no-build-isolation .
install-demo:
	$(PYTHON) -m venv .venv-demo
	$(DEMO_PYTHON) -m pip install -r requirements-demo.txt pytest==9.1.1 httpx==0.28.1
	$(DEMO_PYTHON) -m pip install --no-deps --no-build-isolation .
verify-model:
	$(PYTHON) -m readmit_iq.serving.model_loader
api: verify-model
	$(PYTHON) -m uvicorn readmit_iq.serving.api:app --host 127.0.0.1 --port 8000 --no-access-log
demo:
	STREAMLIT_BROWSER_GATHER_USAGE_STATS=false $(DEMO_PYTHON) -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.headless=true --server.maxUploadSize=1
phase6-test:
	$(PYTHON) -m pytest tests/serving --junitxml=.cache/phase6-tests.xml
phase6: verify-model lint
	$(PYTHON) -m pip check
	READMITIQ_REQUIRE_REAL_MODEL=1 $(PYTHON) -m pytest --junitxml=.cache/phase6-all-tests.xml
	$(DEMO_PYTHON) -m pytest tests/serving/test_demo.py --junitxml=.cache/phase6-ui-tests.xml
	$(PYTHON) scripts/verify_phase6.py
docker-build: verify-model
	docker compose build
docker-run: verify-model
	docker compose up -d --wait
docker-stop:
	docker compose down
