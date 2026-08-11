import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = decodeURIComponent(new URL("../data/FARM HARVEST DATA.xlsx", import.meta.url).pathname);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const targetSheet = workbook.worksheets.getItem("Target Weights");
const previewDirectory = new URL("farm_workbook_previews/", import.meta.url);
await fs.mkdir(previewDirectory, { recursive: true });

const green = "#174F3A";
const pale = "#EEF5EF";
const border = "#C9D8D0";
const checkpoints = new Map([[0, 40], [7, 170], [14, 380], [21, 800], [28, 1200], [35, 1800]]);
const formerSmoothed = [
  40, 51, 65, 82, 101, 124, 150, 180, 203, 228, 257, 288, 322, 360, 400,
  451, 503, 558, 615, 675, 736, 800, 893, 989, 1088, 1188, 1290, 1394, 1500,
  1570, 1640, 1711, 1783, 1854, 1927, 2000,
];
const gompertz = { asymptote: 5564.927652132053, displacement: 4.644852972589512, rate: 0.040266583239551505 };

function linearWeight(day) {
  if (day >= 35) return 1800;
  const anchors = [...checkpoints.entries()];
  for (let index = 0; index < anchors.length - 1; index += 1) {
    const [startDay, startWeight] = anchors[index];
    const [endDay, endWeight] = anchors[index + 1];
    if (day >= startDay && day <= endDay) {
      return Math.round(startWeight + ((day - startDay) / (endDay - startDay)) * (endWeight - startWeight));
    }
  }
  return 40;
}

function scaledWeight(day) {
  if (day >= 35) return 1800;
  const anchors = [...checkpoints.entries()];
  for (let index = 0; index < anchors.length - 1; index += 1) {
    const [startDay, startWeight] = anchors[index];
    const [endDay, endWeight] = anchors[index + 1];
    if (day >= startDay && day <= endDay) {
      const oldStart = formerSmoothed[startDay];
      const oldEnd = formerSmoothed[endDay];
      const progress = (formerSmoothed[day] - oldStart) / (oldEnd - oldStart);
      return Math.round(startWeight + progress * (endWeight - startWeight));
    }
  }
  return 40;
}

const rows = [];
let priorLinear = null;
let priorScaled = null;
for (let day = 0; day <= 49; day += 1) {
  const linear = linearWeight(day);
  const scaled = scaledWeight(day);
  const gompertzWeight = Math.round(gompertz.asymptote * Math.exp(-gompertz.displacement * Math.exp(-gompertz.rate * day)));
  const source = day === 0 ? "Working placement anchor" : checkpoints.has(day) ? "Farm-approved checkpoint" : day < 35 ? "Estimated between checkpoints" : "Day 35 target carried forward";
  rows.push([
    day,
    linear,
    priorLinear === null ? null : linear - priorLinear,
    scaled,
    priorScaled === null ? null : scaled - priorScaled,
    gompertzWeight,
    checkpoints.has(day) ? gompertzWeight - checkpoints.get(day) : null,
    source,
  ]);
  priorLinear = linear;
  priorScaled = scaled;
}

targetSheet.getRange("A1:H51").clear({ contentsOnly: false });
targetSheet.getRange("A1:H1").values = [[
  "Age",
  "Target Weight (Linear Interpolation)",
  "Daily Gain (Linear Interpolation)",
  "Target Weight (Scaled Interpolation)",
  "Daily Gain (Scaled Interpolation)",
  "Gompertz Candidate",
  "Gompertz Error vs Approved Checkpoint",
  "Target Source",
]];
targetSheet.getRange("A2:H51").values = rows;
targetSheet.getRange("A1:H1").format = {
  fill: green,
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  borders: { preset: "all", style: "thin", color: border },
};
targetSheet.getRange("A2:H51").format = {
  borders: { preset: "all", style: "thin", color: "#E5ECE8" },
  verticalAlignment: "center",
};
targetSheet.getRange("A2:H51").conditionalFormats.addCustom("=MOD(ROW(),2)=0", { fill: pale });
targetSheet.getRange("A1:H51").format.autofitColumns();
targetSheet.getRange("A:A").format.columnWidth = 9;
targetSheet.getRange("B:G").format.columnWidth = 23;
targetSheet.getRange("H:H").format.columnWidth = 30;
targetSheet.freezePanes.freezeRows(1);

const sheets = [
  ["Harvest Report (2)", "A1:H20"],
  ["Farm Harvest Data (Daily)", "A1:L20"],
  ["Temperature", "A1:L20"],
  ["Target Weights", "A1:H38"],
  ["Farm Harvest Data (Weekly)", "A1:L20"],
  ["Farm Harvest Data (By Cycle)", "A1:L20"],
];
for (const [sheetName, range] of sheets) {
  const inspection = await workbook.inspect({
    kind: "table",
    range: `'${sheetName}'!${range}`,
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 12,
    summary: `${sheetName} verification`,
  });
  await fs.writeFile(new URL(`${sheetName.replaceAll(" ", "_").replaceAll("/", "-")}.inspect.ndjson`, previewDirectory), inspection.ndjson);
  const image = await workbook.render({ sheetName, range, scale: 0.8, format: "png" });
  await fs.writeFile(new URL(`${sheetName.replaceAll(" ", "_").replaceAll("/", "-")}.png`, previewDirectory), new Uint8Array(await image.arrayBuffer()));
}
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "farm workbook formula error scan",
});
await fs.writeFile(new URL("formula_error_scan.ndjson", previewDirectory), errors.ndjson);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(inputPath);
console.log(`Updated ${inputPath}`);
