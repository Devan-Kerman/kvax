# Contributing Guide

Thank you for your interest in contributing to the project. We welcome contributions and aim to make the process as smooth as possible.

## How to Contribute Code

To contribute code, please follow these steps:

1. **Open an Issue:** Start by [opening an issue](new-issue) to discuss your idea or bug fix before starting work.
2. **Fork the Repository:** Fork the repository and create a new branch to work on your changes.
3. **Submit a Pull Request:** Once you’ve made your changes, submit a Pull Request for review.

## Development Setup

To set up your development environment, you’ll first need to install the following:

- Python 3.10 or later
- Install the [GPU version of JAX](https://jax.readthedocs.io/en/latest/installation.html)

Next, install the required dependencies:

```bash
pip install -e .[dev]
```

**Note: The automatically installed versions of Triton and JAX-Triton might not be compatible. If you encounter an error while running the provided benchmarks, please ensure that you install compatible versions manually. For benchmarking, we used `triton==3.1` and `jax-triton==0.2.0`.**

## Testing

### Writing Tests

Please write or add a unit test that covers the feature you’ve added or the bug you’ve fixed.

### Running Tests

Run the tests locally with the command:

```bash
PYTHONPATH=$PYTHONPATH:$(pwd) pytest ./tests
```

## Code Quality

### Linting

Enable pre-commit hooks to ensure your code meets project standards:

```bash
pre-commit install
```

Alternatively, run the linter manually:

```bash
pre-commit run --all
```
