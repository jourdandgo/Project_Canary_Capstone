import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputPath = new URL("../data/Project_Canary_Revised_Target_Weights.xlsx", import.meta.url);
const previewPath = new URL("target_weights_preview.png", import.meta.url);
const readmePreviewPath = new URL("target_weights_readme_preview.png", import.meta.url);
const workbook = Workbook.create();
const green = "#174F3A";
const lime = "#B8D936";
const pale = "#EEF5EF";
const border = "#C9D8D0";

const checkpoints = [
  [0, 40],
  [7, 170],
  [14, 380],
  [21, 800],
  [28, 1200],
  [35, 1800],
];
const formerSmoothed = [
  40, 51, 65, 82, 101, 124, 150, 180,
  203, 228, 257, 288, 322, 360, 400,
  451, 503, 558, 615, 675, 736, 800,
  893, 989, 1088, 1188, 1290, 1394, 1500,
  1570, 1640, 1711, 1783, 1854, 1927, 2000,
];
const gompertz = { asymptote: 5564.927652132053, displacement: 4.644852972589512, rate: 0.040266583239551505 };

const readme = workbook.worksheets.add("Read Me");
readme.showGridLines = false;
readme.mergeCells("A1:D2");
readme.getRange("A1").values = [["Project Canary · Revised Target Weights"]];
readme.getRange("A1:D2").format = {
  fill: green,
  font: { bold: true, color: "#FFFFFF", size: 18 },
  verticalAlignment: "center",
};
readme.getRange("A4:D4").merge();
readme.getRange("A4").values = [["Working anchor and farm-approved weekly checkpoints"]];
readme.getRange("A4:D4").format = { fill: lime, font: { bold: true, color: green } };
readme.getRange("A5:B11").values = [["Day", "Target (g)"], ...checkpoints];
readme.getRange("A5:B5").format = { fill: green, font: { bold: true, color: "#FFFFFF" } };
readme.getRange("A5:B11").format.borders = { preset: "all", style: "thin", color: border };
readme.getRange("C5:D11").values = [
  ["Status", "Meaning"],
  ["Working anchor", "Former placement-weight assumption"],
  ["Approved", "Doc Raymond checkpoint"],
  ["Approved", "Doc Raymond checkpoint"],
  ["Approved", "Doc Raymond checkpoint"],
  ["Approved", "Doc Raymond checkpoint"],
  ["Approved", "Doc Raymond checkpoint"],
];
readme.getRange("C5:D5").format = { fill: green, font: { bold: true, color: "#FFFFFF" } };
readme.getRange("C5:D11").format.borders = { preset: "all", style: "thin", color: border };
readme.getRange("A13:D13").merge();
readme.getRange("A13").values = [["How Canary fills the days between checkpoints"]];
readme.getRange("A13:D13").format = { fill: lime, font: { bold: true, color: green } };
readme.getRange("A14:D20").merge();
readme.getRange("A14").values = [[
  "Canary uses the checkpoint-calibrated smoothed series: it preserves the former farm curve's proportional within-week shape and meets all revised farm checkpoints exactly. A three-parameter Gompertz curve was also tested because broiler growth is nonlinear. It fit the six anchors with 20 g MAE but missed individual approved checkpoints by as much as 42 g, so it is shown as a scientific comparison—not used as the operating target. Day 35 observed weight remains the model target (Y).",
]];
readme.getRange("A14:D20").format = { fill: pale, wrapText: true, verticalAlignment: "top" };
readme.getRange("A1:D20").format.autofitColumns();
readme.getRange("A:A").format.columnWidth = 20;
readme.getRange("B:B").format.columnWidth = 18;
readme.getRange("C:C").format.columnWidth = 20;
readme.getRange("D:D").format.columnWidth = 38;

const sheet = workbook.worksheets.add("Target Weights");
sheet.showGridLines = false;
const headers = [
  "Age",
  "Target Weight (Linear Interpolation)",
  "Daily Gain (Linear Interpolation)",
  "Former Smoothed Curve Reference",
  "Target Weight (Scaled Interpolation)",
  "Daily Gain (Scaled Interpolation)",
  "Gompertz Candidate",
  "Gompertz Error vs Approved Checkpoint",
];
sheet.getRange("A1:H1").values = [headers];
sheet.getRange("A1:H1").format = {
  fill: green,
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  borders: { preset: "all", style: "thin", color: border },
};
sheet.getRange("A2:A51").values = Array.from({ length: 50 }, (_, index) => [index]);
sheet.getRange("D2:D51").values = Array.from({ length: 50 }, (_, index) => [
  index <= 35 ? formerSmoothed[index] : 1800,
]);

function linearFormula(row) {
  const age = `A${row}`;
  return `=ROUND(IF(${age}<=7,40+${age}/7*(170-40),IF(${age}<=14,170+(${age}-7)/7*(380-170),IF(${age}<=21,380+(${age}-14)/7*(800-380),IF(${age}<=28,800+(${age}-21)/7*(1200-800),IF(${age}<=35,1200+(${age}-28)/7*(1800-1200),1800))))),0)`;
}

function smoothFormula(row) {
  const age = `A${row}`;
  const ref = `D${row}`;
  return `=ROUND(IF(${age}<=7,40+(${ref}-$D$2)/($D$9-$D$2)*(170-40),IF(${age}<=14,170+(${ref}-$D$9)/($D$16-$D$9)*(380-170),IF(${age}<=21,380+(${ref}-$D$16)/($D$23-$D$16)*(800-380),IF(${age}<=28,800+(${ref}-$D$23)/($D$30-$D$23)*(1200-800),IF(${age}<=35,1200+(${ref}-$D$30)/($D$37-$D$30)*(1800-1200),1800))))),0)`;
}

sheet.getRange("B2:B51").formulas = Array.from({ length: 50 }, (_, index) => [linearFormula(index + 2)]);
sheet.getRange("C2").values = [[null]];
sheet.getRange("C3:C51").formulas = Array.from({ length: 49 }, (_, index) => [`=B${index + 3}-B${index + 2}`]);
sheet.getRange("E2:E51").formulas = Array.from({ length: 50 }, (_, index) => [smoothFormula(index + 2)]);
sheet.getRange("F2").values = [[null]];
sheet.getRange("F3:F51").formulas = Array.from({ length: 49 }, (_, index) => [`=E${index + 3}-E${index + 2}`]);
sheet.getRange("G2:G51").values = Array.from({ length: 50 }, (_, index) => [
  Math.round(gompertz.asymptote * Math.exp(-gompertz.displacement * Math.exp(-gompertz.rate * index))),
]);
const checkpointMap = new Map(checkpoints);
sheet.getRange("H2:H51").values = Array.from({ length: 50 }, (_, index) => [
  checkpointMap.has(index) ? Math.round(gompertz.asymptote * Math.exp(-gompertz.displacement * Math.exp(-gompertz.rate * index)) - checkpointMap.get(index)) : null,
]);
sheet.getRange("A2:H51").format = {
  borders: { preset: "all", style: "thin", color: "#E5ECE8" },
  verticalAlignment: "center",
};
sheet.getRange("A2:H51").conditionalFormats.addCustom("=MOD(ROW(),2)=0", { fill: pale });
sheet.getRange("A1:H51").format.autofitColumns();
sheet.getRange("A:A").format.columnWidth = 10;
sheet.getRange("B:H").format.columnWidth = 24;
sheet.freezePanes.freezeRows(1);
sheet.tables.add("A1:H51", true, "RevisedTargetWeights");

const preview = await workbook.render({ sheetName: "Target Weights", range: "A1:H38", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const readmePreview = await workbook.render({ sheetName: "Read Me", range: "A1:D20", scale: 1, format: "png" });
await fs.writeFile(readmePreviewPath, new Uint8Array(await readmePreview.arrayBuffer()));
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "revised target-weight formula error scan",
});
console.log(errors.ndjson);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath.pathname);
console.log(`Saved ${outputPath.pathname}`);
