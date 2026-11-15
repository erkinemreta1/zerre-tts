// Node harness for web/public/phonemize-tr.js.
// Usage: node web_phonemes.js input.json > output.json
// input.json: {"sentences": ["...", ...], "lexicon": "path", "symbols": [...]}
const fs = require("fs");
const path = require("path");

const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const lexicon = JSON.parse(fs.readFileSync(input.lexicon, "utf8"));
globalThis.fetch = undefined;
require(path.resolve("web/public/phonemize-tr.js"));
const phonemize = globalThis.zerre.phonemize;

const out = input.sentences.map((sentence) => phonemize(sentence, lexicon, input.symbols));
process.stdout.write(JSON.stringify(out));
