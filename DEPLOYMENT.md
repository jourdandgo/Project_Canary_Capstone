# Project Canary — GitHub and Streamlit Deployment

## Repository contents

This repository contains the Streamlit application, versioned rules, trained prototype model artifacts, tests, methodology, and defense documentation.

The capstone repository intentionally bundles the current daily farm workbook and final-weight summary so the dashboard opens with preliminary results. This should be done only with the farm owner's approval, especially when the repository or Streamlit app is public.

## Run locally

```bash
uv sync --dev
uv run streamlit run app.py
```

The app loads its bundled files automatically. To test a newer data cut, optionally upload:

1. `FARM HARVEST DATA.xlsx` using **Update daily farm data (optional)**.
2. `Farm Performance Summary.xlsx` using **Update final-weight data (optional)**.

The second workbook is used only for the final average-weight field. Harvest recovery continues to be calculated from the daily farm workbook.

## Deploy on Streamlit Community Cloud

1. Push this folder as the root of a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Select `app.py` as the entrypoint.
4. Select Python 3.12 in advanced settings.
5. Deploy the app.
6. Confirm that the bundled dashboard opens; use the sidebar only when demonstrating a newer workbook.

The repository uses `uv.lock` and `pyproject.toml` for reproducible Python dependencies.

## Privacy boundary

- Do not add any workbooks beyond the two explicitly approved bundled capstone files, and never commit credentials or `.streamlit/secrets.toml`.
- A public app may expose uploaded farm results to anyone using that session. Use a private app or sanitized demonstration workbook when confidentiality matters.
- The in-app rule editor writes to the running application's local filesystem. Treat edits on Community Cloud as demonstration-session changes rather than durable production administration.

## Before presenting

```bash
uv run pytest
uv run python -m scripts.validate_capstone
```

Confirm that the app opens, all seven pages load, the six building cards render, and both workbook upload controls work.
