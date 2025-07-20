# Installation

This guide provides all the necessary information to install `CulicidaeLab`. We cover a simple installation for regular users and a complete development setup for those who wish to contribute to the project.

### Prerequisites

Before you begin, please ensure you have the following installed on your system:

- **Python 3.11 or higher**
- **pip** (Python's package installer, usually included with Python)
- **Git** (for the development setup)

## Standard Installation (For Users)

This is the recommended approach for most users who want to use `CulicidaeLab` in their own projects. It will install the latest stable version from the Python Package Index (PyPI).

We highly recommend working within a **virtual environment** to avoid conflicts with other projects or system-wide packages.

### Recommended: Using `uv`

`uv` is an extremely fast, modern Python package installer and resolver that can replace `pip` and `venv`.

```bash
# 1. Create and activate a virtual environment
uv venv

# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 2. Install the library
uv add culicidaelab
```

### Alternative: Using `pip` and `venv`

If you prefer to use the standard tools built into Python:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Install the library using pip
pip install culicidaelab
```

## Development Setup (For Contributors)

If you plan to contribute to `CulicidaeLab`, fix a bug, or add a new feature, you will need to set up a development environment. This involves cloning the repository and installing the project in **editable mode**.

1.  **Fork the Repository**
    Start by forking the [main repository](https://github.com/iloncka-ds/culicidaelab) on GitHub to your own account.

2.  **Clone Your Fork**
    Clone your forked repository to your local machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/culicidaelab.git
    cd culicidaelab
    ```

3.  **Create and Activate a Virtual Environment**
    A virtual environment is essential for development to isolate your dependencies.
    ```bash
    # Using uv (recommended for speed)
    uv venv

    # Or using Python's built-in venv
    # python -m venv .venv

    # Activate the environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    # .venv\Scripts\activate
    ```

4.  **Install Dependencies in Editable Mode**
    Install the project with the `[dev]` extra, which includes all tools needed for testing, linting, and documentation. The `-e` flag installs it in "editable" mode, meaning changes you make to the source code will be immediately effective without needing to reinstall.

    ```bash
    # This command installs the library and all development dependencies
    uv pip install -e ".[dev]"
    ```

5.  **Set Up Pre-commit Hooks**
    We use `pre-commit` to automatically run code quality checks before each commit. This is a one-time setup step per project clone.
    ```bash
    pre-commit install
    ```
    Now, whenever you run `git commit`, our code formatters and linters will run automatically, ensuring your contributions match the project's standards.

## Verifying the Installation

To ensure that `CulicidaeLab` was installed correctly, you can run the following Python code snippet:

```python
try:
    from culicidaelab import get_settings
    settings = get_settings()
    print("✅ CulicidaeLab installation successful!")
    print(f"Default model directory: {settings.model_dir}")
except ImportError:
    print("❌ CulicidaeLab installation failed. Please check the steps above.")
```

You're all set! Now you're ready to explore the library's features.
