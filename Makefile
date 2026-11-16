.PHONY: help install test lint audit demo build clean status

help:
	@echo "install  - editable install with dev extras"
	@echo "test     - run the test suite"
	@echo "lint     - ruff check"
	@echo "audit    - verify the phoneme inventory against eSpeak NG"
	@echo "demo     - phonemize one Turkish sentence"
	@echo "status   - rebuild docs/status.html from the real state"
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

status:
	uv run python scripts/status_report.py

taste:
	uv run python scripts/taste.py

smoke:
	uv run python -m zerre.train --train data/tr/annotations/train.jsonl \
	  --validation data/tr/annotations/validation.jsonl --output runs/smoke \
	  --device cpu --subset 8 --steps 6 --batch-frames 3000 --max-batch 4 \
	  --gan-start 100 --teacher-hold 2 --teacher-ramp 4 --save-every 3 --log-every 1 \
	  --max-seconds 900

build:
	uv build

clean:
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
