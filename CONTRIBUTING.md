# Contributing to MolToken

First off, thank you for considering contributing to MolToken! It's people like you that make MolToken such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots if relevant

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Include screenshots and animated GIFs in your pull request whenever possible
* Follow the Python styleguides
* Include thoughtfully-worded, well-structured tests
* Document new code
* End all files with a newline

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Styleguide

* All Python code must adhere to the PEP 8 style guide
* Use Black for code formatting
* Use type hints for function arguments and return values
* Use descriptive variable names
* Include docstrings for all public methods and classes

### Documentation Styleguide

* Use Markdown for documentation
* Reference function and class names using backticks
* Include code examples in documentation when relevant
* Keep line length to a maximum of 80 characters
* Include section headers in bold

## Setting Up Development Environment

1. Fork and clone the repository
2. Create a new conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate tokenmol
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[test]"
   ```
4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tokenizer.py

# Run with coverage report
pytest --cov=moltoken tests/
```

## Making a Release

1. Update version number in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. Build and upload to PyPI:
   ```bash
   python -m build
   twine upload dist/*
   ```

## Questions?

Feel free to open an issue or contact the maintainers if you have any questions.

Thank you for your contributions! 🎉