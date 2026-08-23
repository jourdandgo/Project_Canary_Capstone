"""Build the clean Model 1 and Model 3 reconstructions for Canary's trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canary.three_model_evaluation import build_legacy_reconstructions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../capstone_FINAL_v18"))
    parser.add_argument("--output", type=Path, default=Path("models/three_model/legacy"))
    parser.add_argument("--workbook", type=Path, default=Path("data/FARM HARVEST DATA.xlsx"))
    args = parser.parse_args()
    result = build_legacy_reconstructions(args.source, args.output, args.workbook)
    summary = {
        model_id: {
            "status": item["status"],
            "mae": item["selected_metrics"]["mae"],
            "cycle_macro_mae": item["selected_metrics"]["cycle_macro_mae"],
            "audit_mae": item["audit_metrics"]["mae"],
            "thi_features": item["thi_features"],
        }
        for model_id, item in result["models"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
