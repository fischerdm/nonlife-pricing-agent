# Non Life Pricing Agent

## Setup

**Prerequisites:** [pyenv](https://github.com/pyenv/pyenv) must be installed to manage the Python version. On macOS: `brew install pyenv`. For other platforms see the [pyenv installation docs](https://github.com/pyenv/pyenv#installation).

Pin the Python version for this project (creates a `.python-version` file):

```bash
pyenv local 3.12.8
```

Create and activate the virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```