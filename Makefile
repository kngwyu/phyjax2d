.PHONY: test lint

test:
	uv run pytest

lint:
	uvx ruff check
	uvx black src/phyjax2d tests --check
	uvx isort src/phyjax2d tests --check


format:
	uvx black src/phyjax2d tests
	uvx isort src/phyjax2d tests


publish:
	uv build
	uvx twine upload dist/*

all: test lint
