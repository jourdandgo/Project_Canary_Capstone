#!/usr/bin/env python3
"""Run the isolated Project Canary farm-wide optimization round."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canary.bodyweight_modeling_review import SEED
from canary.model_optimization_round import run_optimization_round


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=ROOT / "data" / "FARM HARVEST DATA.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "farmwide_model_finalization_round")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--profile", choices=("balanced", "full", "smoke"), default="full")
    parser.add_argument("--audit-cycle", default="latest")
    arguments = parser.parse_args()
    manifest = run_optimization_round(
        arguments.workbook,
        arguments.output,
        seed=arguments.seed,
        profile=arguments.profile,
        audit_cycle=arguments.audit_cycle,
    )
    print(f"Completed {manifest['round_version']} at {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
