import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const outputDir = path.join(root, "outputs", "model_ready");
const payloadPath = path.join(outputDir, "model_ready_payload.json");
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));

const workbook = Workbook.create();
const green = "#173F31";
const green2 = "#286245";
const lime = "#C7F24B";
const soft = "#EDF5F0";
const line = "#D7E4DC";
const muted = "#607069";

function columnName(index) {
  let value = index + 1;
  let name = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    value = Math.floor((value - 1) / 26);
  }
  return name;
}

function normalizeValue(column, value) {
  if (value == null) return null;
  if (column.endsWith("_date") || column === "as_of_date") {
    const date = new Date(`${value}T12:00:00Z`);
    return Number.isNaN(date.getTime()) ? value : date;
  }
  return value;
}

function titleFor(sheetName) {
  const titles = {
    "How to Read": "How to Read the Model-Ready Data",
    "Data Dictionary": "Data Dictionary",
    "Building Outcomes": "One Row per Building Outcome",
    "Recovery Training": "Exact Recovery Training Snapshots",
    "Recovery Schedule": "Why Recovery Uses Standard and Latest Snapshots",
    "Weight Training": "Exact Day 35 Weight Training Checkpoints",
    "Latest Cycle Weight Audit": "Latest-Cycle Day 35 Prospective Audit",
    "Latest Recovery Audit": "Latest-Cycle Recovery Prospective Audit",
    "Recovery Daily Audit": "All Leakage-Safe Recovery Candidate Snapshots",
  };
  return titles[sheetName] || sheetName;
}

function noteFor(sheetName) {
  const notes = {
    "How to Read": "Start here. This sheet explains what one row means, which fields are X versus Y, and why the recovery and weight tables have different snapshot schedules.",
    "Data Dictionary": "Definitions, units, X/Y roles, missing-value handling, and leakage controls for every exported field.",
    "Building Outcomes": `Outcome-level audit view: ${payload.summary.recovery_building_outcomes} historical fitting outcomes plus ${payload.summary.latest_cycle_recovery_audit_candidates} later 2026-3 audit candidates. Temperature and humidity change by review date and belong in the as-of snapshots.`,
    "Recovery Training": `The ${payload.summary.recovery_training_rows.toLocaleString()} balanced rows used for recovery model comparison: standard Days 7, 14, 21, and 28 plus a separately labelled latest pre-outcome snapshot. A Day 48 row is a near-end snapshot, not an extra standard checkpoint.`,
    "Recovery Schedule": "This table counts the exact days retained. Standard checkpoints answer how forecasting changes with age; the fifth training snapshot tests near-end forecasting and can vary from Day 23 to Day 48.",
    "Weight Training": `The ${payload.summary.weight_training_rows.toLocaleString()} historical X snapshots at Days 7, 14, 21, and 28. The recorded Day 35 weight is the Y target on every row; it is intentionally never an X feature.`,
    "Latest Cycle Weight Audit": `The ${payload.summary.latest_cycle_day35_audit_rows.toLocaleString()} latest-cycle checkpoint rows are a prospective check only. They are excluded from model fitting and champion selection.`,
    "Latest Recovery Audit": `The frozen recovery model was scored on ${payload.summary.latest_cycle_recovery_audit_rows.toLocaleString()} later 2026-3 checkpoints. Its MAE is ${payload.summary.latest_cycle_recovery_audit_mae_pp.toFixed(2)} percentage points; treat this as a provisional audit because the endpoint is still a last-recorded-population proxy.`,
    "Recovery Daily Audit": `All ${payload.summary.recovery_daily_audit_rows.toLocaleString()} candidate daily snapshots before balancing. Each row uses only records available by its as-of date.`,
  };
  return notes[sheetName] || "Project Canary model-ready evidence.";
}

function csvText(records) {
  if (!records.length) return "";
  const headers = Object.keys(records[0]);
  const escape = (value) => {
    if (value == null) return "";
    const text = String(value);
    return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [
    headers.map(escape).join(","),
    ...records.map((record) => headers.map((header) => escape(record[header])).join(",")),
  ].join("\n") + "\n";
}

for (const [sheetIndex, [sheetName, records]] of Object.entries(payload.tables).entries()) {
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  const headers = records.length ? Object.keys(records[0]) : ["No data"];
  const lastColumn = columnName(headers.length - 1);
  const titleLastColumn = headers.length > 13 ? "M" : lastColumn;
  const lastRow = records.length + 5;

  sheet.getRange(`A1:${titleLastColumn}1`).merge();
  sheet.getRange("A1").values = [[titleFor(sheetName)]];
  sheet.getRange(`A1:${titleLastColumn}1`).format = {
    fill: green,
    font: { bold: true, color: "#FFFFFF", size: 16 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${titleLastColumn}1`).format.rowHeight = 30;

  sheet.getRange(`A2:${titleLastColumn}2`).merge();
  sheet.getRange("A2").values = [[noteFor(sheetName)]];
  sheet.getRange(`A2:${titleLastColumn}2`).format = {
    fill: soft,
    font: { color: green2, italic: true, size: 10 },
    wrapText: true,
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${titleLastColumn}2`).format.rowHeight = 34;

  sheet.getRange(`A3:${titleLastColumn}3`).merge();
  sheet.getRange("A3").values = [[
    `Source: ${payload.summary.source_workbook} · Generated from ${payload.summary.canonical_building_day_rows.toLocaleString()} canonical building-day rows · ${records.length.toLocaleString()} rows on this sheet`,
  ]];
  sheet.getRange(`A3:${titleLastColumn}3`).format = {
    font: { color: muted, size: 9 },
  };

  sheet.getRange(`A5:${lastColumn}5`).values = [headers];
  if (records.length) {
    const rows = records.map((record) => headers.map((header) => normalizeValue(header, record[header])));
    sheet.getRange(`A6:${lastColumn}${lastRow}`).values = rows;
    const tableName = `CanaryTable${sheetIndex + 1}`;
    const table = sheet.tables.add(`A5:${lastColumn}${lastRow}`, true, tableName);
    table.style = "TableStyleMedium4";
    table.showBandedRows = true;
    table.showFilterButton = true;
  }
  sheet.getRange(`A5:${lastColumn}5`).format = {
    fill: green2,
    font: { bold: true, color: "#FFFFFF", size: 9 },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: line },
  };
  sheet.getRange(`A5:${lastColumn}5`).format.rowHeight = 34;
  if (records.length) {
    sheet.getRange(`A6:${lastColumn}${lastRow}`).format = {
      font: { color: "#2A3730", size: 9 },
      verticalAlignment: "center",
    };
    for (let col = 0; col < headers.length; col += 1) {
      const header = headers[col];
      const letter = columnName(col);
      const values = records.slice(0, 100).map((record) => String(record[header] ?? ""));
      const longest = Math.max(header.length, ...values.map((value) => value.length));
      const descriptive = ["plain_language_definition", "missing_value_handling", "leakage_guard"].includes(header);
      const width = descriptive ? 34 : Math.min(Math.max(longest + 2, 11), 24);
      sheet.getRange(`${letter}:${letter}`).format.columnWidth = width;
      if (descriptive) sheet.getRange(`${letter}6:${letter}${lastRow}`).format.wrapText = true;
      if (header.endsWith("_date") || header === "as_of_date") {
        sheet.getRange(`${letter}6:${letter}${lastRow}`).format.numberFormat = "yyyy-mm-dd";
      } else if (header.includes("recovery") || header === "percentage_alive" || header === "cumulative_mortality_rate") {
        sheet.getRange(`${letter}6:${letter}${lastRow}`).format.numberFormat = "0.00%";
      } else if (header.endsWith("_kg") || header.endsWith("_kg_y")) {
        sheet.getRange(`${letter}6:${letter}${lastRow}`).format.numberFormat = "0.000";
      } else if (header.endsWith("_g") || header.includes("population") || header.includes("inventory")) {
        sheet.getRange(`${letter}6:${letter}${lastRow}`).format.numberFormat = "#,##0";
      }
    }
  }
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(sheetName === "Data Dictionary" ? 2 : 2);
}

await fs.mkdir(outputDir, { recursive: true });
await fs.writeFile(
  path.join(outputDir, "recovery_training.csv"),
  csvText(payload.tables["Recovery Training"]),
  "utf8",
);
await fs.writeFile(
  path.join(outputDir, "day35_weight_training.csv"),
  csvText(payload.tables["Weight Training"]),
  "utf8",
);
await fs.writeFile(
  path.join(outputDir, "latest_cycle_day35_audit.csv"),
  csvText(payload.tables["Latest Cycle Weight Audit"]),
  "utf8",
);
await fs.writeFile(
  path.join(outputDir, "latest_cycle_recovery_audit.csv"),
  csvText(payload.tables["Latest Recovery Audit"]),
  "utf8",
);
await fs.writeFile(
  path.join(outputDir, "model_ready_manifest.json"),
  JSON.stringify(payload.summary, null, 2) + "\n",
  "utf8",
);

for (const sheetName of Object.keys(payload.tables)) {
  const preview = await workbook.render({
    sheetName,
    range: sheetName === "Data Dictionary" ? "A1:H24" : "A1:M18",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(outputDir, `preview_${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const inspect = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 7000,
  tableMaxRows: 4,
  tableMaxCols: 10,
});
await fs.writeFile(path.join(outputDir, "workbook_inspection.ndjson"), inspect.ndjson, "utf8");
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(outputDir, "formula_error_scan.ndjson"), errors.ndjson, "utf8");

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, "Project_Canary_Model_Ready_Data.xlsx"));
console.log(JSON.stringify(payload.summary, null, 2));
