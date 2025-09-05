# Contributing to CulicidaeLab

First off, thank you for considering contributing to `CulicidaeLab`! We are thrilled you're here and appreciate your interest in making this project better. Every contribution, from a small typo fix to a major new feature, is valuable.

This document provides guidelines for contributing to the project. Please read it carefully to ensure a smooth and effective collaboration process.

## Code of Conduct

This project and everyone participating in it is governed by the [CulicidaeLab Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [iloncka.ds@gmail.com](mailto:iloncka.ds@gmail.com).

## How Can I Contribute?

There are many ways to contribute, and all of them are welcome:

*   **Reporting Bugs:** If you find a bug, please open an issue and provide detailed information, including steps to reproduce it.
*   **Suggesting Enhancements:** Have an idea for a new feature or an improvement to an existing one? Open an issue to start a discussion.
*   **Writing Documentation:** Help us improve our documentation by fixing typos, clarifying confusing sections, or adding new examples.
*   **Submitting Pull Requests:** If you're ready to contribute code, this is the way to go.

### Reporting Bugs

Before creating a bug report, please check the [existing issues](https://github.com/iloncka-ds/culicidaelab/issues) to see if the problem has already been reported. If it hasn't, please open a new issue and include the following:

*   A clear and descriptive title.
*   The version of `culicidaelab` you are using.
*   Your operating system and Python version.
*   A step-by-step description of how to reproduce the bug.
*   A code snippet that demonstrates the issue.
*   The full traceback of any error messages.

### Suggesting Enhancements

We'd love to hear your ideas for improving `CulicidaeLab`. To suggest an enhancement, please open an issue and provide:

*   A clear and descriptive title.
*   A detailed explanation of the proposed enhancement and why it would be beneficial.
*   (Optional) A rough sketch of how the feature might be implemented or used in code.

## Development Setup

Ready to write some code? Here’s how to set up your development environment.

1.  **Fork the Repository**
    Start by forking the [main repository](https://github.com/iloncka-ds/culicidaelab) on GitHub.

2.  **Clone Your Fork**
    Clone your forked repository to your local machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/culicidaelab.git
    cd culicidaelab
    ```

3.  **Create a Virtual Environment**
    It is highly recommended to work in a virtual environment. You can use `uv` or Python's built-in `venv` module.
    ```bash
    # Using uv (recommended)
    uv venv

    # Or using venv
    python -m venv .venv
    ```
    Activate the environment:
    ```bash
    # On macOS/Linux
    source .venv/bin/activate

    # On Windows
    .venv\Scripts\activate
    ```

4.  **Install Dependencies**
    Install the project in editable mode (`-e`) along with all development dependencies (`[dev]`).
    ```bash
    uv pip install -e ".[dev]"
    ```
    This command installs everything you need for development, including testing tools, linters, and documentation generators.

5.  **Install pre-commit Hooks**
    We use `pre-commit` to automatically run linters and formatters before each commit. This ensures code quality and consistency across the project.
    ```bash
    pre-commit install
    ```
    This is a one-time setup per clone. Now, whenever you run `git commit`, the hooks defined in `.pre-commit-config.yaml` will be executed.

## Our Development Workflow

We use a suite of tools to maintain code quality. Your `pre-commit` setup handles all of this automatically, but it's good to know what's happening under the hood.

*   **Formatting:** `black` and `ruff-format` are used for deterministic, consistent code formatting.
*   **Linting:** `ruff` and `flake8` catch common programming errors and style issues.
*   **Type Checking:** `mypy` performs static type checking to find type-related bugs before runtime.
*   **Security:** `bandit` scans for common security vulnerabilities in the code.
*   **Modernization:** `pyupgrade` automatically upgrades syntax to newer Python versions.

### Running Tests

To ensure your changes haven't introduced any regressions, please run the full test suite using `pytest`.
```bash
pytest tests
```
All tests should pass before you submit a pull request.

### Writing Documentation

Good documentation is as important as good code. If you add or modify a feature, please update the documentation in the `docs/` directory accordingly.

You can preview your changes locally by running:
```bash
mkdocs serve --config-file=mkdocs.en.yml --clean
# or
mkdocs serve --config-file=mkdocs.ru.yml --clean
```
This will start a local server, and you can view the documentation site in your browser at `http://127.0.0.1:8000/culicidaelab/en` and `http://127.0.0.1:8000/culicidaelab/ru` corresponding to the English and Russian versions.

## Submitting a Pull Request

When you're ready to submit your changes, please follow these steps:

1.  **Create a New Branch**
    Create a descriptive branch for your changes from the `main` branch.
    ```bash
    git checkout -b feature/your-awesome-feature
    ```

2.  **Make Your Changes**
    Write your code, add or update tests, and write documentation.

3.  **Commit Your Work**
    Commit your changes with a clear and concise message. When you commit, the `pre-commit` hooks will run. If they fail, fix the reported issues and commit again.
    ```bash
    git add .
    git commit -m "feat: Add awesome new feature"
    ```

4.  **Push to Your Fork**
    Push your branch to your forked repository on GitHub.
    ```bash
    git push origin feature/your-awesome-feature
    ```

5.  **Open a Pull Request**
    Go to the `CulicidaeLab` repository on GitHub and open a pull request.
    *   Provide a clear title and a detailed description of your changes.
    *   If your PR addresses an existing issue, link to it by including `Closes #123` in the description.

6.  **Code Review**
    Once your PR is submitted, a maintainer will review it. We may suggest some changes or improvements. We will do our best to provide timely and constructive feedback.

Thank you again for your interest in contributing! We look forward to your submissions.
