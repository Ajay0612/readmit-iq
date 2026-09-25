PYTHON ?= .venv/bin/python
RUFF = .venv/bin/ruff

.PHONY: install download inspect notebook verify test lint phase1 eda eda-notebook phase2 split baselines baseline-notebook phase3
.PHONY: optimize optimization-reports optimization-notebook phase4
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
	$(PYTHON) -m pytest --junitxml=.cache/phase4-tests.xml
	$(PYTHON) scripts/verify_phase4.py
