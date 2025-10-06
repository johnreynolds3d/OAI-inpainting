# Contributing Guide

Thank you for your interest in contributing to the OAI Inpainting project! This guide will help you get started with development and ensure your contributions meet our standards.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-compatible GPU (recommended)

### Getting Started

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/OAI-inpainting.git
   cd OAI-inpainting
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev,ml]"
   pre-commit install
   ```

## Code Quality Standards

This project uses modern Python development tools for maintaining high code quality:

### Ruff (Linting & Formatting)

- **Replaces**: Black + flake8 + isort
- **Purpose**: Fast Python linter and formatter
- **Configuration**: `pyproject.toml`

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix issues automatically
ruff check . --fix
```

### MyPy (Type Checking)

- **Purpose**: Static type checking
- **Configuration**: `pyproject.toml`

```bash
# Run type checking
mypy src/ scripts/
```

### Pre-commit Hooks

Automatically run on every commit:
- Code formatting with Ruff
- Linting checks
- YAML/JSON validation
- Large file detection
- Line ending fixes

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the existing code style
- Add type hints where appropriate
- Write clear, descriptive commit messages
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 4. Check Code Quality

```bash
# Format and lint
ruff format .
ruff check . --fix

# Type checking
mypy src/ scripts/

# All checks (what pre-commit runs)
pre-commit run --all-files
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style Guidelines

### Python Code

- **Line length**: 88 characters (enforced by Ruff)
- **Type hints**: Use type hints for function parameters and return values
- **Docstrings**: Use Google-style docstrings for public functions/classes
- **Imports**: Sort imports with Ruff (isort-compatible)

### Example Code Style

```python
from typing import List, Optional, Union
from pathlib import Path

def process_images(
    image_paths: List[Path],
    output_dir: Path,
    quality: int = 95
) -> Optional[bool]:
    """
    Process a list of images and save to output directory.

    Args:
        image_paths: List of input image paths
        output_dir: Directory to save processed images
        quality: JPEG quality (1-100)

    Returns:
        True if successful, None if error
    """
    # Implementation here
    pass
```

### Configuration Files

- Use YAML for configuration files
- Keep configurations in `configs/` directory
- Use descriptive names and comments

### Documentation

- Update README.md for user-facing changes
- Update docstrings for code changes
- Add examples for new features

## Testing Guidelines

### Unit Tests

- Test individual functions and classes
- Use descriptive test names
- Mock external dependencies
- Aim for high coverage

### Integration Tests

- Test complete workflows
- Use real data when possible
- Test error conditions

### Test Structure

```python
def test_function_name_should_do_something():
    """Test that function_name does something specific."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_name(input_data)

    # Assert
    assert result == expected_output
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Type checking passes
- [ ] Documentation updated
- [ ] Commit messages are clear

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Getting Help

- **Issues**: Create an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check existing docs in `docs/`

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag
4. Update documentation

---

Thank you for contributing to the OAI Inpainting project! 🎉
