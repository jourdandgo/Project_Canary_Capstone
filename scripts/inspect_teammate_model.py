"""Safely inventory a trusted teammate notebook/model before fair Canary evaluation.

The pickle is never executed. This script records its checksum and pickle opcode
inventory; the reproducible notebook approach must still be rebuilt and evaluated
with Canary's complete-cycle holdouts before it can become a champion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_notebook(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    lowered = code.lower()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "code_cells": sum(cell.get("cell_type") == "code" for cell in notebook.get("cells", [])),
        "grouped_cycle_validation_terms": [
            term for term in ("leaveonegroupout", "groupkfold", "groups=") if term in lowered
        ],
        "random_row_split_terms": [
            term for term in ("train_test_split", "kfold(", "cross_val_score") if term in lowered
        ],
        "model_terms": [
            term for term in ("ridge", "linearregression", "randomforest", "xgboost", "gradientboosting") if term in lowered
        ],
        "warning": "Terms are a screening aid, not proof of leakage safety or reproducibility.",
    }


def inspect_pickle(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    globals_found: list[str] = []
    opcodes: dict[str, int] = {}
    parse_error = None
    try:
        for opcode, argument, _ in pickletools.genops(data):
            opcodes[opcode.name] = opcodes.get(opcode.name, 0) + 1
            if opcode.name in {"GLOBAL", "INST"} and argument is not None:
                globals_found.append(str(argument))
    except Exception as exc:  # static inspection must report malformed artifacts safely
        parse_error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": len(data),
        "pickle_globals": sorted(set(globals_found)),
        "opcode_counts": opcodes,
        "parse_error": parse_error,
        "executed": False,
        "warning": "Pickle files can execute code when loaded. Load only after trust and reproducibility review.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/teammate_model_inventory.json"))
    args = parser.parse_args()
    report = {"notebook": inspect_notebook(args.notebook)}
    if args.model:
        report["model"] = inspect_pickle(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
