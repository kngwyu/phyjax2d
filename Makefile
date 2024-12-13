.PHONY: test lint

CUDA_AVAILABLE := $(shell command -v nvcc >/dev/null 2>&1 && echo 1 || echo 0)

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


sync:
ifeq ($(CUDA_AVAILABLE),1)
	uv sync --all-extras
else
	uv sync --extra=vis
endif


all: test lint
