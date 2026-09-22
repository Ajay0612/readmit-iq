PYTHON ?= .venv/bin/python
RUFF = .venv/bin/ruff

.PHONY: install download inspect notebook verify test lint phase1
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
