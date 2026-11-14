.PHONY: test test-fast lint format check run

# The whole suite with coverage. Third-party services are stubbed; needs local Postgres with pgvector.
test:
	uv run pytest --cov --cov-report=term-missing

# Same tests, no coverage, stop at the first failure.
test-fast:
	uv run pytest -x -q

lint:
	uv run ruff check backend scripts

format:
	uv run ruff check --fix backend/app scripts

# Lint and the whole suite: what CI runs.
check: lint test

run:
	uv run uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
