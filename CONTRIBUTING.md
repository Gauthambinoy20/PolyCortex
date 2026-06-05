# Contributing

Thanks for your interest in contributing!

## Development setup

```bash
git clone <repo>
cd PolymarketTrader
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Fork and create a feature branch from `prod-hardening` (or `main`).
2. Write tests first when fixing bugs or adding features.
3. Run the suite:
   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   pytest --cov=src/polymarket_agent --cov-fail-under=85
   ```
4. Open a PR against `prod-hardening` — fill in the PR template.

## Style

- Conventional Commits for commit messages (e.g. `feat(execution): add OCO`).
- One logical change per commit when possible.
- No secrets, no `print()` in library code (use `logging`).
- Type hints required for all public functions.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## Security

Never commit private keys.  Report vulnerabilities privately — see
[SECURITY.md](SECURITY.md).
