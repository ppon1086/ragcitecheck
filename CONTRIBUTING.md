# Contributing

Thanks for your interest in contributing to `ragcitecheck`.

## Development setup

Clone the repo and create a virtual environment.

### Windows

```bash
py -3.13 -m venv .venv313
.venv313\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
pip install pytest
```

### macOS / Linux

```bash
python3.13 -m venv .venv313
source .venv313/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
pip install pytest
```

## Running tests

Run the smoke tests with:

```bash
pytest -q
```

## Project scope

`ragcitecheck` focuses on evidence stability diagnostics for RAG pipelines, including:

- document-level citation overlap
- span-level evidence overlap
- per-query instability summaries
- lightweight validation and reporting workflows

Please keep changes aligned with this scope.

## Pull requests

Please open pull requests against the `dev` branch.

A good PR should:

- keep changes focused and small
- include tests when behavior changes
- avoid committing generated outputs or large artifacts
- update docs when CLI behavior or input format changes

## Repo hygiene

Please do not commit:

- `__pycache__`
- `.egg-info`
- local virtual environments
- generated reports
- benchmark output folders
- zip archives
- large experiment artifacts

## Reporting issues

If you open an issue, it is helpful to include:

- the command you ran
- the input format used
- the exact error message
- Python version
- operating system
