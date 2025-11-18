# Contributing to Triton-Augment

Thank you for your interest in contributing to Triton-Augment! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in your interactions with other contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/triton-augment.git
   cd triton-augment
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/originalowner/triton-augment.git
   ```

## Development Setup

### Prerequisites

- Python 3.8+, PyTorch 2.0+, Triton 2.0+
- CUDA-capable GPU

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
pytest tests/
```

**Alternative**: Use [uv](https://github.com/astral-sh/uv) for faster installs: `uv pip install -e ".[dev]"`

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, PyTorch version, GPU model)
- Any relevant code snippets or error messages

### Suggesting Enhancements

We welcome feature requests! Please open an issue with:

- A clear description of the feature
- The motivation/use case for the feature
- Any implementation ideas you have

### Contributing Code

We welcome code contributions! Here are some areas where you can help:

1. **New Transforms**: Implement additional image transformations (see Roadmap in README)
2. **Performance Optimizations**: Improve existing kernel implementations
3. **Bug Fixes**: Fix reported issues
4. **Documentation**: Improve docs, examples, and tutorials
5. **Tests**: Add more comprehensive test coverage

## Pull Request Process

1. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our [coding standards](#coding-standards)

3. **Add tests** for your changes (if applicable)

4. **Run the test suite** to ensure everything passes:
   ```bash
   pytest tests/
   ```

5. **Format your code**:
   ```bash
   black triton_augment/
   isort triton_augment/
   ```

6. **Commit your changes** with a clear commit message:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub with:
   - A clear title and description
   - Reference to any related issues
   - Description of what was changed and why
   - Any breaking changes or migration notes

### PR Review Process

- All PRs require at least one review before merging
- Automated tests must pass
- Code must follow the project's coding standards
- Documentation must be updated if applicable

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 100)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints where appropriate

### Triton Kernels

- Add comprehensive docstrings explaining kernel parameters and behavior
- Use descriptive variable names
- Include comments for complex operations
- Optimize for memory coalescence and bank conflicts

### Naming Conventions

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: prefix with `_`

### Documentation

- All public functions and classes must have Google-style docstrings
- Include type hints and usage examples
- See existing code for docstring format

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=triton_augment --cov-report=html

# Run specific test file
pytest tests/test_transforms.py

# Run specific test
pytest tests/test_transforms.py::TestFunctionalAPI::test_apply_brightness
```

### Writing Tests

- Place tests in `tests/` directory with descriptive names
- Test normal cases and edge cases
- Ensure tests are deterministic (set random seeds)
- Follow existing test structure in the codebase

## Documentation

### Building Documentation

```bash
cd docs/
make html
```

### Documentation Guidelines

- Keep README.md up to date
- **Important**: README.md and docs/index.md are separate (optimized for GitHub and MkDocs respectively). Update both if making major changes.
- Add docstrings to all public APIs
- Include usage examples in docstrings
- Update CHANGELOG.md for significant changes

## Performance Benchmarking

When adding new features or optimizing existing code, please include benchmark results:

```bash
python examples/benchmark.py
python examples/benchmark_video.py
```

Include benchmark results in your PR description, comparing before and after performance.

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the "question" label
- Start a discussion in the Discussions tab
- Reach out to the maintainers

---

Thank you for contributing to Triton-Augment! Your efforts help make GPU-accelerated image augmentation accessible to everyone.

