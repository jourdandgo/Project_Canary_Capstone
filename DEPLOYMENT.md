# Project Canary — GitHub and Streamlit Deployment

## Repository contents

This repository contains the Streamlit application, versioned rules, trained prototype model artifacts, tests, methodology, and defense documentation.

It intentionally excludes the farm's raw workbooks. Do not commit those files unless the farm owner has explicitly approved publication.

## Run locally

```bash
uv sync --dev
uv run streamlit run app.py
```

Then upload:

1. `FARM HARVEST DATA.xlsx` using **Farm workbook**.
2. `Farm Performance Summary.xlsx` using **Final-weight workbook (optional)** when completed-cycle final average weights are needed.

The second workbook is used only for the final average-weight field. Harvest recovery continues to be calculated from the daily farm workbook.

## Deploy on Streamlit Community Cloud

1. Push this folder as the root of a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Select `app.py` as the entrypoint.
4. Select Python 3.12 in advanced settings.
5. Deploy the app.
6. Upload the workbooks through the sidebar when demonstrating the application.

The repository uses `uv.lock` and `pyproject.toml` for reproducible Python dependencies.

## Privacy boundary

- Do not commit raw farm workbooks, credentials, or `.streamlit/secrets.toml`.
- A public app may expose uploaded farm results to anyone using that session. Use a private app or sanitized demonstration workbook when confidentiality matters.
- The in-app rule editor writes to the running application's local filesystem. Treat edits on Community Cloud as demonstration-session changes rather than durable production administration.

## Before presenting

```bash
uv run pytest
uv run python -m scripts.validate_capstone
```

Confirm that the app opens, all seven pages load, the six building cards render, and both workbook upload controls work.
