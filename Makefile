.PHONY: help install test lint audit demo build clean

help:
	@echo "install  - editable install with dev extras"
	@echo "test     - run the test suite"
	@echo "lint     - ruff check"
	@echo "audit    - verify the phoneme inventory against eSpeak NG"
	@echo "demo     - phonemize one Turkish sentence"
	@echo "build    - wheel + sdist"

install:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts

audit:
	uv run python scripts/audit_inventory.py

demo:
	uv run python -c "from zerre.text import phonemize; print(' '.join(phonemize('Merhaba dünya!').symbols))"

build:
	uv build

clean:
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
