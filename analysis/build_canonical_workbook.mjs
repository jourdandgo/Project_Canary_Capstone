import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = new URL("../", import.meta.url);
const inputPath = new URL("canonical_workbook_inputs.json", import.meta.url);
const outputPath = new URL("../data/Project_Canary_Canonical_Building_Day_1666.xlsx", import.meta.url);
const previewDir = new URL("canonical_workbook_previews/", import.meta.url);
const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));

function excelColumn(index) {
  let value = index + 1;
  let output = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    output = String.fromCharCode(65 + remainder) + output;
    value = Math.floor((value - 1) / 26);
  }
  return output;
}

const workbook = Workbook.create();
const green = "#174F3A";
const lime = "#B8D936";
const pale = "#EEF5EF";
const border = "#C9D8D0";

function addDataSheet(name, headers, rows, tableName) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const endColumn = excelColumn(headers.length - 1);
  const endRow = rows.length + 1;
  sheet.getRange(`A1:${endColumn}${endRow}`).values = [headers, ...rows];
  sheet.getRange(`A1:${endColumn}1`).format = {
    fill: green,
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: border },
  };
  sheet.getRange(`A2:${endColumn}${endRow}`).format = {
    borders: { preset: "all", style: "thin", color: "#E5ECE8" },
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${endColumn}${endRow}`).format.autofitColumns();
  sheet.getRange(`A1:${endColumn}${Math.min(endRow, 100)}`).format.autofitRows();
  for (let column = 0; column < headers.length; column += 1) {
    const range = sheet.getRange(`${excelColumn(column)}1:${excelColumn(column)}${endRow}`);
    const nameLower = String(headers[column]).toLowerCase();
    if (nameLower.includes("evidence") || nameLower.includes("warning")) {
      range.format.columnWidth = 40;
      range.format.wrapText = true;
    } else if (nameLower.includes("date")) {
      range.format.columnWidth = 13;
    } else {
      range.format.columnWidth = Math.min(Math.max(12, String(headers[column]).length + 2), 24);
    }
  }
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(Math.min(3, headers.length));
  sheet.tables.add(`A1:${endColumn}${endRow}`, true, tableName);
  return sheet;
}

const readme = workbook.worksheets.add("Read Me");
readme.showGridLines = false;
readme.mergeCells("A1:D2");
readme.getRange("A1").values = [["Project Canary · Canonical Building-Day Dataset"]];
readme.getRange("A1:D2").format = {
  fill: green,
  font: { bold: true, color: "#FFFFFF", size: 18 },
  verticalAlignment: "center",
};
readme.mergeCells("A4:D4");
readme.getRange("A4").values = [["Purpose"]];
readme.getRange("A4:D4").format = { fill: lime, font: { bold: true, color: green } };
readme.mergeCells("A5:D7");
readme.getRange("A5").values = [[
  "This workbook is the auditable output of Canary's ingestion process. It consolidates repeated source rows into exactly one record per harvest cycle, building, and production age. The original FARM HARVEST DATA.xlsx remains the source of truth and is not modified.",
]];
readme.getRange("A5:D7").format = { wrapText: true, verticalAlignment: "top", fill: pale };

const q = payload.quality;
const summaryRows = [
  ["Source rows", q.source_rows],
  ["Unique canonical building-days", q.canonical_rows],
  ["Building-days with repeated source rows", q.duplicate_keys],
  ["Extra source rows consolidated", q.duplicate_rows_consolidated],
  ["Production-value conflicts", q.production_conflict_keys],
  ["Weight measurement days", q.weight_measurement_days],
  ["Zone-aggregated building-days", q.zone_aggregated_days],
  ["Maximum environment sections", q.maximum_environment_sections],
  ["Temperature coverage", q.temperature_coverage_pct / 100],
  ["Humidity coverage", q.humidity_coverage_pct / 100],
];
readme.getRange("A9:B9").values = [["Quality check", "Result"]];
readme.getRange("A10:B19").values = summaryRows;
readme.getRange("A9:B9").format = { fill: green, font: { bold: true, color: "#FFFFFF" } };
readme.getRange("A9:B19").format.borders = { preset: "all", style: "thin", color: border };
readme.getRange("B18:B19").format.numberFormat = "0.0%";

readme.getRange("A21:D21").merge();
readme.getRange("A21").values = [["How repeated rows were handled"]];
readme.getRange("A21:D21").format = { fill: lime, font: { bold: true, color: green } };
readme.getRange("A22:D26").merge();
readme.getRange("A22").values = [[
  "The 119 repeated building-days contain identical production values but multiple Zone A / Zone B environmental readings. Canary keeps the first available production value, takes the minimum of section minimums, maximum of section maximums, and an unweighted mean of section averages. The Duplicate Audit sheet preserves all 238 source rows so the consolidation can be checked directly.",
]];
readme.getRange("A22:D26").format = { wrapText: true, verticalAlignment: "top", fill: pale };
readme.getRange("A28:D28").values = [["Sheet", "What it contains", "Rows", "Use"]];
readme.getRange("A29:D33").values = [
  ["Canonical Building-Day", "One normalized record per cycle-building-age", q.canonical_rows, "Risk, forecasting, EDA, and dashboard input"],
  ["Duplicate Summary", "Counts of repeated days by cycle and building", payload.duplicate_summary.rows.length, "Quick duplicate review"],
  ["Duplicate Audit", "All source rows participating in repeated keys", payload.duplicate_rows.rows.length, "Trace source values and consolidation"],
  ["Daily Target Curve", "Doc Raymond's checkpoints plus estimated daily targets", payload.targets.rows.length, "Age-specific weight comparisons"],
  ["Weight Checkpoints", "Availability of Day 7/14/21/28/35 measurements", payload.checkpoint_coverage.rows.length, "Audit model-label coverage"],
];
readme.getRange("A28:D28").format = { fill: green, font: { bold: true, color: "#FFFFFF" } };
readme.getRange("A28:D33").format.borders = { preset: "all", style: "thin", color: border };
readme.getRange("A1:D33").format.autofitColumns();
readme.getRange("A:A").format.columnWidth = 34;
readme.getRange("B:B").format.columnWidth = 58;
readme.getRange("C:C").format.columnWidth = 14;
readme.getRange("D:D").format.columnWidth = 44;
readme.freezePanes.freezeRows(2);

addDataSheet(
  "Canonical Building-Day",
  payload.canonical.headers,
  payload.canonical.rows,
  "CanonicalBuildingDayTable",
);
addDataSheet(
  "Duplicate Summary",
  payload.duplicate_summary.headers,
  payload.duplicate_summary.rows,
  "DuplicateSummaryTable",
);
addDataSheet(
  "Duplicate Audit",
  payload.duplicate_rows.headers,
  payload.duplicate_rows.rows,
  "DuplicateAuditTable",
);
addDataSheet(
  "Daily Target Curve",
  payload.targets.headers,
  payload.targets.rows,
  "DailyTargetCurveTable",
);
addDataSheet(
  "Weight Checkpoints",
  payload.checkpoint_coverage.headers,
  payload.checkpoint_coverage.rows,
  "WeightCheckpointCoverageTable",
);

await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, range] of [
  ["Read Me", "A1:D33"],
  ["Canonical Building-Day", "A1:H18"],
  ["Duplicate Summary", "A1:F10"],
  ["Duplicate Audit", "A1:T12"],
  ["Daily Target Curve", "A1:F37"],
  ["Weight Checkpoints", "A1:G18"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(new URL(`${safeName}.png`, previewDir), new Uint8Array(await preview.arrayBuffer()));
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "canonical workbook formula-error scan",
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath.pathname);
console.log(`Saved ${outputPath.pathname}`);
