.PHONY: smoke test train benchmark clean

# End-to-end offline smoke: load data, train 3 regressors, exercise predict + Flask.
smoke:
	python scripts/smoke.py

# Unit tests.
test:
	python -m pytest -q

# Train the default (ridge) model and dump artifacts to ./models.
train:
	python -m src.train --config configs/default.yaml

# Benchmark all four models on a fixed split.
benchmark:
	python -m src.benchmark

# Remove caches and trained artifacts.
clean:
	rm -rf models/*.pkl __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
