.PHONY: train test fmt

VENV ?= .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip        # <- add this for clarity

$(VENV):
	python -m venv $(VENV) 
	$(PIP) install -r requirements.txt  # installs inside that venv

train: $(VENV)
	$(PYTHON) src/train.py --config configs/default.yaml

test: $(VENV)
	pytest -q

fmt:
	$(VENV)/bin/ruff check --fix src
