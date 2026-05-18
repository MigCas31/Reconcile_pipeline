.PHONY: format lint typecheck test verify

format:
	ruff format .

lint:
	ruff check --fix .

typecheck:
	mypy reconcile/ reconcile_v2/

test:
	python3 -m pytest tests/ -v

verify: lint format typecheck test
	@echo "All checks passed"
