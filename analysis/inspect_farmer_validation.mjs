import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/jourdan.go/Downloads/PROJECT CANARY/Farmer Validation Workbook.xlsx";
const outputDir = new URL("farmer_validation_previews/", import.meta.url);
await fs.mkdir(outputDir, { recursive: true });

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 20000,
  tableMaxRows: 30,
  tableMaxCols: 12,
  tableMaxCellChars: 120,
});
await fs.writeFile(new URL("inspection.ndjson", outputDir), overview.ndjson, "utf8");

for (const sheet of workbook.worksheets.items) {
  const used = sheet.getUsedRange();
  if (!used) continue;
  const preview = await workbook.render({
    sheetName: sheet.name,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const safeName = sheet.name.replace(/[^A-Za-z0-9_-]+/g, "_");
  await fs.writeFile(
    new URL(`${safeName}.png`, outputDir),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

console.log(overview.ndjson);
