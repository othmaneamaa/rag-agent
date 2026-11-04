# Raccourcis pour Linux et macOS ; sous Windows, lancez les commandes à la main.
.PHONY: install corpus index eval seuil test qualite api ui roue

install:
	pip install -r requirements.txt -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
	pip install -e .

corpus:
	python src/telecharger.py

index:
	python src/ingest.py

eval:
	python src/evaluate.py

seuil:
	python src/courbe_seuil.py

test:
	python -m pytest -q

qualite:
	ruff check .
	mypy rag_agent api

api:
	rag serve

ui:
	rag ui

roue:
	rm -rf dist && python -m build --wheel
