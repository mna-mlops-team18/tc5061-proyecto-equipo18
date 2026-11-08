#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tc5061-proyecto-equipo18
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python
# Ambiente aparte de la validacion de datos (DeepChecks no importa en Python 3.14)
VALIDACION_PY ?= .venv-validacion/bin/python
# Imagen de la API: podman si esta instalado, si no docker (make image CONTAINER=docker)
CONTAINER ?= $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)
IMAGEN ?= mibici-api:latest
PUERTO ?= 8000

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download, normalize and version the MiBici data (ingesta) with the MLflow Project
.PHONY: data
data:
	mlflow run . -e ingest --env-manager local

## Verify local data against references/manifiesto_datos.json
.PHONY: verify
verify:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo18.dataset verify

## Build the P1, P2 and P3 datasets with the MLflow Project
.PHONY: features
features:
	mlflow run . -e features --env-manager local

## Start the local MLflow tracking server (UI + Model Registry) on http://127.0.0.1:5000
.PHONY: mlflow-server
mlflow-server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

## Train one model with the MLflow Project, e.g. make train PREGUNTA=p1 MODELO=hgb_poisson
.PHONY: train
train:
	mlflow run . -e train -P pregunta=$(PREGUNTA) -P modelo=$(MODELO) --env-manager local

## Run the nested hyperparameter search of one question, e.g. make search PREGUNTA=p2
.PHONY: search
search:
	mlflow run . -e search -P pregunta=$(PREGUNTA) --env-manager local

## Register the chosen model of one question with the champion/challenger aliases, e.g. make register PREGUNTA=p1
.PHONY: register
register:
	mlflow run . -e register -P pregunta=$(PREGUNTA) --env-manager local

## Export the champion of P1 from the Model Registry to models/servicio/p1 for the API image
.PHONY: export-model
export-model:
	$(PYTHON_INTERPRETER) -m tc5061_proyecto_equipo18.modeling.exportar --pregunta p1

## Serve the API locally with reload on http://127.0.0.1:8000/docs (needs models/servicio)
.PHONY: api
api:
	uvicorn api.main:app --reload --port $(PUERTO)

## Build the API image from the Containerfile (needs the models exported to models/servicio)
.PHONY: image
image:
	@test -f models/servicio/p1/MLmodel || { echo "Falta models/servicio/p1: exporta los modelos primero (README > Despliegue de la API)"; exit 1; }
	$(CONTAINER) build -f Containerfile -t $(IMAGEN) .

## Run the API image on http://127.0.0.1:8000 (Ctrl+C to stop)
.PHONY: run-api
run-api:
	$(CONTAINER) run --rm --name mibici-api -p 127.0.0.1:$(PUERTO):8000 $(IMAGEN)

## Run the tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest

## Create the separate DeepChecks environment (Python 3.12, needs uv)
.PHONY: venv-validacion
venv-validacion:
	uv venv .venv-validacion --python 3.12
	uv pip install --python $(VALIDACION_PY) -r requirements-validacion.txt

## DeepChecks data validation in its own environment (needs data/processed; fails if missing)
.PHONY: validate
validate:
	VALIDACION_REQUIERE_DATOS=1 $(VALIDACION_PY) -m pytest validacion_datos -rs

## Security checks: known vulnerabilities (pip-audit), code analysis (bandit) and repo secrets
.PHONY: security
security:
	pip-audit --skip-editable --strict
	pip-audit --strict -r requirements-api.txt
	bandit -r tc5061_proyecto_equipo18 $(wildcard validacion_datos) -x validacion_datos/test_validacion.py
	$(PYTHON_INTERPRETER) -m pytest tests/test_seguridad.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
