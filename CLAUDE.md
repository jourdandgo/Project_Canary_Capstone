# Project Canary — Claude Code Entry Point

Read `docs/HANDOFF.md` before making any change.

This folder is the GitHub-ready mirror of the canonical Streamlit app in `../canary_app/`. Use it for distribution or GitHub work, but implement and validate changes in the canonical app first, then mirror targeted approved files here. Never overwrite unrelated changes, and do not commit, push, reset, or delete unless the user explicitly asks.

The project is an explainable early-warning and decision-support prototype for JJ Agriventures—not a diagnosis tool, automated treatment system, causal engine, or guaranteed forecasting system. Its central story is:

**See earlier → investigate earlier → learn every cycle.**

When the complete parent workspace is present, the detailed authoritative handoff is in:

- `../CLAUDE.md`
- `../canary_app/docs/PROJECT_CONTEXT.md`
- `../canary_app/docs/CURRENT_STATE.md`

If this directory is used standalone, use `docs/HANDOFF.md`, then inspect the included `docs/`, `config/risk_rules.json`, and `models/trish_v18/manifest.json` before acting.
