.PHONY: train test fmt

VENV ?= .venv
PYTHON = $(VENV)/bin/python

$(VENV):
	python -m venv $(VENV) && $(VENV)/bin/pip install -r requirements.txt

train: $(VENV)
	$(PYTHON) src/train.py --config configs/default.yaml

test: $(VENV)
	pytest -q

fmt:
	$(VENV)/bin/ruff check --fix src
