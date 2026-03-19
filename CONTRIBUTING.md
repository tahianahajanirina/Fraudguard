# Contributing to FraudGuard

Thank you for your interest in contributing to FraudGuard! This guide will help you get started.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/Fraudguard.git
   cd Fraudguard
   ```
3. **Create a branch** from `develop`:
   ```bash
   git checkout -b feature/your-feature develop
   ```
4. **Set up** the environment:
   ```bash
   cp .env.example .env
   make up
   ```

## Development Workflow

### Branch Naming

- `feature/description` — New features
- `fix/description` — Bug fixes
- `docs/description` — Documentation
- `refactor/description` — Code refactoring

### Code Style

We use **Ruff** for linting and formatting (configured in `pyproject.toml`):

```bash
make lint      # Check for issues
make format    # Auto-format code
```

Rules: line length 100, Python 3.11, rule sets E + F + I.

### Testing

Run tests before submitting:

```bash
make test                # All tests
make test-api            # API tests only
make test-preprocessing  # Preprocessing tests
make test-model          # Model tests
make test-pipeline       # Pipeline tests
```

Tests run inside Docker containers to match the production environment.

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new endpoint for model comparison
fix: correct scaler loading path
docs: update API reference
test: add batch prediction edge cases
ci: update GitHub Actions workflow
refactor: simplify model loading logic
```

## Submitting Changes

1. **Push** your branch:
   ```bash
   git push origin feature/your-feature
   ```
2. **Open a Pull Request** against `develop`
3. **Fill in** the PR template
4. **Wait for review** — maintainers will review your changes

## Reporting Issues

- Use the **Bug Report** template for bugs
- Use the **Feature Request** template for new ideas
- Include steps to reproduce, expected behavior, and actual behavior

## Project Structure

See the [README](README.md) for a full project structure overview and the [docs/](docs/) folder for detailed architecture documentation.

## Questions?

Open an issue with the `question` label, and we'll be happy to help.
