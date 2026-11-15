// Runs the model (ONNX Runtime Web) off the main thread and phonemizes Turkish
// with the lexicon built by scripts/build_lexicon.py plus a rule fallback.
// Messages in: {type:"load"}, {type:"speak", id, text, voice, speed}.
// Messages out: status, ready, sentence, done, error.

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
importScripts(ORT + "ort.wasm.min.js");

let session, config, symbols, symbolIndex, lexicon;
const VERSION = "1";

const status = (text, progress) => postMessage({ type: "status", text, progress });

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

async function download(url, label) {
  const key = `${url}?v=${VERSION}`;
  let store = null;
  try {
    store = await caches.open("zerre-tts-" + VERSION);
    const hit = await store.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (error) {
    store = null;
  }
  const bytes = await fetchBytes(key, label);
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

// The words of the text, lowercased the way the lexicon builder does it.
function words(text) {
  return text.toLocaleLowerCase("tr").match(/[\p{L}]+/gu) || [];
}

// Rule fallback, mirroring scripts/build_lexicon.py's letter mapping.
const VOWEL_PHONE = new Set(["a", "e", "i", "o", "u", "y", "ø", "œ", "ɯ"]);
const BACK = new Set(["a", "ı", "o", "u"]);
const BASE = { a: "a", b: "b", c: "dʒ", ç: "tʃ", d: "d", e: "e", f: "f", g: "g", h: "h",
  ı: "ɯ", i: "i", j: "ʒ", k: "k", l: "l", m: "m", n: "n", o: "o", ö: "œ", p: "p",
  r: "ɾ", s: "s", ş: "ʃ", t: "t", u: "u", ü: "y", v: "v", y: "j", z: "z" };

function fallbackPhones(word) {
  const phones = [];
  const stress = [];
  for (let index = 0; index < word.length; index++) {
    const letter = word[index];
    if (letter === "ğ") {
      if (phones.length && /[aeıioöuü]/.test(word[index - 1])) phones[phones.length - 1] += "ː";
      continue;
    }
    if (!(letter in BASE)) continue;
    const following = word[index + 1];
    const front = following && /[eiöü]/.test(following);
    let phone = BASE[letter];
    if (letter === "k" && front) phone = "c";
    if (letter === "g" && front) phone = "ɟ";
    if (letter === "l" && word[index - 1] && BACK.has(word[index - 1])) phone = "ɫ";
    phones.push(phone);
    stress.push(0);
  }
  for (let index = phones.length - 1; index >= 0; index--) {
    if (VOWEL_PHONE.has(phones[index].replace("ː", ""))) {
      stress[index] = 1;
      break;
    }
  }
  return { phones, stress };
}

function phonemize(text) {
  const ids = [];
  const stress = [];
  const push = (symbol) => {
    const id = symbolIndex.get(symbol);
    if (id === undefined) return;
    ids.push(id);
    stress.push(0);
  };
  push("<bos>");
  let lastWasWord = false;
  for (const token of text.match(/[\p{L}]+|[.,!?;:—…]/gu) || []) {
    if (/^[\p{L}]+$/u.test(token)) {
      if (lastWasWord) push("<sp>");
      const key = token.toLocaleLowerCase("tr");
      const entry = lexicon.words[key];
      let phones, levels;
      if (entry) {
        phones = entry[0].split(" ");
        levels = entry[1].split("").map(Number);
      } else {
        ({ phones, stress: levels } = fallbackPhones(key));
      }
      phones.forEach((phone, index) => {
        const id = symbolIndex.get(phone);
        if (id === undefined) return;
        ids.push(id);
        stress.push(levels[index] || 0);
      });
      lastWasWord = true;
    } else if (token !== "'") {
      push(token);
      lastWasWord = false;
    }
  }
  push("<eos>");
  return { ids, stress, text: ids.map((id) => symbols[id]).join(" ") };
}

const PEAK = 0.99;
function limitPeak(samples) {
  let loudest = 0;
  for (let index = 0; index < samples.length; index++) loudest = Math.max(loudest, Math.abs(samples[index]));
  if (loudest > PEAK) for (let index = 0; index < samples.length; index++) samples[index] *= PEAK / loudest;
  return samples;
}

async function synthesize(phones, voice, speed) {
  const count = phones.ids.length;
  const integers = (values) =>
    new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids),
    stress: integers(phones.stress),
    voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voice)]), [1]),
    speed: scalar(speed),
    pitch_shift: scalar(0),
    pitch_range: scalar(1),
  });
  return output.audio.data;
}

async function speak({ id, text, voice, speed }) {
  const started = performance.now();
  const sentences = text.replace(/\s+/g, " ").trim().split(/(?<=[.!?…])\s+/).filter(Boolean);
  const chunks = [];
  let phonemeTime = 0;
  let modelTime = 0;
  for (let index = 0; index < sentences.length; index++) {
    const mark = performance.now();
    const phones = phonemize(sentences[index]);
    phonemeTime += performance.now() - mark;
    if (phones.ids.length < 3 || phones.ids.length > config.max_tokens) continue;
    const before = performance.now();
    const audio = await synthesize(phones, voice, speed);
    modelTime += performance.now() - before;
    chunks.push(audio);
    const heard = limitPeak(audio.slice());
    postMessage({ type: "sentence", id, index, total: sentences.length, phonemes: phones.text,
      audio: heard, sampleRate: config.sample_rate }, [heard.buffer]);
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: "Metin okunamadı.", plain: true });
    return;
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    audio.set(chunk, offset);
    offset += chunk.length;
  }
  postMessage({ type: "done", id, audio: limitPeak(audio), sampleRate: config.sample_rate,
    phonemeTime, modelTime, totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      status("Sözlük yükleniyor");
      lexicon = await (await fetch("lexicon-tr.json")).json();
      status("Model yükleniyor");
      config = await (await fetch("model.json")).json();
      symbols = lexicon.symbols;
      symbolIndex = new Map(symbols.map((symbol, index) => [symbol, index]));
      const bytes = await download("model.onnx", "Model yükleniyor");
      ort.env.wasm.wasmPaths = ORT;
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      postMessage({ type: "ready", voices: config.voices, sampleRate: config.sample_rate });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String((error && error.message) || error) });
  }
};
