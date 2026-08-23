"""Generate the source-backed 2026-3 defense-demo checkpoint bundle."""

from pathlib import Path

from canary.data import load_workbook
from canary.demo import write_demo_bundle


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    dataset = load_workbook(ROOT / "data" / "FARM HARVEST DATA.xlsx")
    manifest = write_demo_bundle(dataset, ROOT / "demo_data" / "2026-3")
    print(f"Created {len(manifest['entries'])} checkpoint CSVs.")
