# Development Workflow Guide

This document outlines the development workflow, standards, and best practices for contributing to the Pixelis project.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Quality Standards](#code-quality-standards)
3. [Testing Requirements](#testing-requirements)
4. [Git Workflow](#git-workflow)
5. [Pull Request Process](#pull-request-process)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Code Review Guidelines](#code-review-guidelines)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Conda (recommended) or virtualenv

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pixelis.git
   cd pixelis
   ```

2. **Create a virtual environment:**
   ```bash
   conda create -n pixelis python=3.10
   conda activate pixelis
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install dev dependencies
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

   This ensures code quality checks run automatically before each commit.

## Code Quality Standards

### Automated Code Quality Enforcement

We use several tools to maintain consistent code quality:

- **Black**: Code formatting (100 char line length)
- **Ruff**: Fast Python linting (replaces flake8, isort, and more)
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning

### Pre-Commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Update hooks to latest versions
pre-commit autoupdate
```

### Code Style Guidelines

1. **Type Hints**: All functions must have type hints:
   ```python
   def process_data(input_path: Path, batch_size: int = 32) -> Dict[str, Any]:
       """Process data from the given path."""
       pass
   ```

2. **Docstrings**: Use Google-style docstrings:
   ```python
   def train_model(config: TrainingConfig) -> ModelOutput:
       """Train the model with given configuration.
       
       Args:
           config: Training configuration object
       
       Returns:
           ModelOutput containing trained model and metrics
       
       Raises:
           ValueError: If configuration is invalid
       """
       pass
   ```

3. **Imports**: Organized automatically by Ruff:
   - Standard library
   - Third-party packages
   - Local imports

4. **Naming Conventions**:
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private methods: `_leading_underscore`

## Testing Requirements

### Test Coverage Policy

**MANDATORY**: Code coverage for core modules (`core/modules/` and `core/engine/`) must maintain **≥70% coverage**.

### Test Organization

```
tests/
├── unit/           # Unit tests (isolated components)
├── integration/    # Integration tests (component interactions)
├── fixtures/       # Shared test fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

1. **Test Naming**: Use descriptive names
   ```python
   def test_artifact_manager_singleton_pattern():
       """Test that ArtifactManager follows singleton pattern."""
       pass
   ```

2. **Test Structure**: Follow Arrange-Act-Assert pattern
   ```python
   def test_model_training():
       # Arrange
       config = create_test_config()
       model = Model(config)
       
       # Act
       result = model.train()
       
       # Assert
       assert result.loss < 1.0
       assert result.accuracy > 0.8
   ```

3. **Markers**: Use appropriate test markers
   ```python
   @pytest.mark.slow
   @pytest.mark.gpu
   def test_full_training_pipeline():
       pass
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov-report=html

# Run specific test types
pytest -m "not slow"  # Skip slow tests
pytest -m unit        # Only unit tests
pytest -m integration # Only integration tests

# Run specific test file
pytest tests/unit/test_artifact_manager.py

# Run with verbose output
pytest -v

# Run in parallel (faster)
pytest -n auto
```

### Test Quality Requirements

**IMPORTANT**: Test coverage is **necessary but not sufficient** for code approval.

Tests must:
- ✅ Have meaningful assertions (not just `assert True`)
- ✅ Cover edge cases and error conditions
- ✅ Test actual behavior, not implementation details
- ✅ Be independent and reproducible
- ✅ Run quickly (mark slow tests appropriately)

## Git Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Urgent production fixes
- `experiment/*`: Experimental features

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build/tool changes
- `perf`: Performance improvements

Example:
```
feat(reproducibility): add artifact versioning system

Implement content-addressable storage for artifacts with SHA-256 hashing.
This ensures immutability and enables exact reproducibility.

Closes #123
```

## Pull Request Process

### Before Creating a PR

1. **Update from main:**
   ```bash
   git checkout main
   git pull origin main
   git checkout your-branch
   git rebase main
   ```

2. **Run all checks locally:**
   ```bash
   # Format and lint
   pre-commit run --all-files
   
   # Run tests
   pytest --cov=core
   
   # Type checking
   mypy core scripts
   ```

3. **Update documentation** if needed

### PR Requirements

- [ ] Descriptive title following commit convention
- [ ] Clear description of changes
- [ ] Tests for new functionality
- [ ] Documentation updates
- [ ] All CI checks passing
- [ ] Code coverage ≥70% for core modules
- [ ] No security vulnerabilities

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No new warnings
```

## CI/CD Pipeline

### Continuous Integration

Our CI pipeline runs on every push and PR:

1. **Code Quality Checks**
   - Pre-commit hooks
   - Linting and formatting
   - Type checking

2. **Testing**
   - Unit tests (Python 3.10, 3.11)
   - Integration tests
   - Coverage reporting

3. **Security Scanning**
   - Dependency vulnerability checks
   - Code security analysis

4. **Reproducibility Check**
   - Artifact system validation
   - Environment capture verification

### Pipeline Optimization

- **Dependency caching**: Dependencies cached based on requirements.txt hash
- **Parallel execution**: Tests run in parallel when possible
- **Matrix testing**: Multiple Python versions tested simultaneously

### Monitoring CI Status

```bash
# Check workflow status
gh workflow view

# View recent runs
gh run list

# Watch a running workflow
gh run watch
```

## Code Review Guidelines

### For Authors

1. **Keep PRs focused**: One feature/fix per PR
2. **Provide context**: Explain why, not just what
3. **Respond promptly**: Address feedback quickly
4. **Test thoroughly**: Include edge cases
5. **Document changes**: Update relevant docs

### For Reviewers

Code reviews must evaluate:

1. **Correctness**: Does the code work as intended?
2. **Design**: Is the approach appropriate?
3. **Testing**: Are tests comprehensive and meaningful?
4. **Performance**: Any performance implications?
5. **Security**: Any security concerns?
6. **Maintainability**: Is the code clean and readable?

### Review Checklist

- [ ] **Implementation Quality**
  - Follows design patterns
  - No code duplication
  - Appropriate abstractions
  - Error handling

- [ ] **Test Quality**
  - Tests are meaningful (not just for coverage)
  - Edge cases covered
  - Assertions validate behavior
  - Tests are maintainable

- [ ] **Documentation**
  - Functions have docstrings
  - Complex logic is commented
  - README updated if needed
  - API changes documented

- [ ] **Performance**
  - No obvious bottlenecks
  - Efficient algorithms used
  - Resource cleanup handled

- [ ] **Security**
  - No hardcoded secrets
  - Input validation present
  - No SQL injection risks
  - Dependencies are safe

### Approval Requirements

- Minimum 1 approval required
- All CI checks must pass
- No unresolved comments
- Coverage threshold met

## Future Optimizations

As the project grows, we plan to:

1. **Parallelize test execution** across multiple CI jobs
2. **Implement incremental testing** (only test changed code)
3. **Add performance regression testing**
4. **Set up GPU testing** on self-hosted runners
5. **Implement visual regression testing** for UI components
6. **Add automated dependency updates**
7. **Implement security scanning** in container images

## Getting Help

- **Documentation**: See `/docs` folder
- **Issues**: File bugs/features on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Contributing**: See `CONTRIBUTING.md`

## Quick Reference

```bash
# Setup
pre-commit install          # Install hooks
pip install -e ".[dev]"     # Install dev deps

# Development
pre-commit run --all-files  # Run all checks
pytest --cov=core          # Run tests with coverage
mypy core scripts          # Type check

# Git
git checkout -b feature/x  # Create feature branch
git commit -m "feat: x"    # Commit with convention
git push origin feature/x  # Push branch

# CI/CD
gh workflow view           # Check CI status
gh run list               # View recent runs
```

---

*This document is a living guide. Please keep it updated as our practices evolve.*