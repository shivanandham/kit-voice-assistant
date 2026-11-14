.PHONY: test test-fast lint run

# The whole suite with coverage. Third-party services are stubbed; needs local Postgres with pgvector.
test:
	uv run pytest --cov --cov-report=term-missing

# Same tests, no coverage, stop at the first failure.
test-fast:
	uv run pytest -x -q

lint:
	uv run ruff check backend scripts

run:
	uv run uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
