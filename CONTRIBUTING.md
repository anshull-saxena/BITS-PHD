# Contributing

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

```bash
make lint
make test
```

Integration tests use synthetic data and tiny training epochs (<2 min).
Full reproduction is marked `@pytest.mark.slow` and not run in CI.
