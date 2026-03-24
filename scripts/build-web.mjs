import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(".");
const sourceDir = resolve(root, "web");
const outputDir = resolve(root, "dist");

if (!existsSync(sourceDir)) {
  throw new Error(`Missing web source directory: ${sourceDir}`);
}

rmSync(outputDir, { recursive: true, force: true });
mkdirSync(outputDir, { recursive: true });
cpSync(sourceDir, outputDir, { recursive: true });
