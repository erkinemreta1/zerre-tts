// Runs the model (ONNX Runtime Web) off the main thread and phonemizes Turkish
// with the lexicon built by scripts/build_lexicon.py plus the rule fallback in
// phonemize-tr.js. Messages in: {type:"load"}, {type:"speak", id, text, voice, speed}.
// Messages out: status, ready, sentence, done, error.

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
importScripts(ORT + "ort.wasm.min.js", "phonemize-tr.js");

let session, config, symbols, lexicon;
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
    const phones = globalThis.zerre.phonemize(sentences[index], lexicon, symbols);
    phonemeTime += performance.now() - mark;
    if (phones.ids.length < 3 || phones.ids.length > config.max_tokens) continue;
    const before = performance.now();
    const audio = await synthesize(phones, voice, speed);
    modelTime += performance.now() - before;
    chunks.push(audio);
    const heard = limitPeak(audio.slice());
    postMessage({ type: "sentence", id, index, total: sentences.length,
      phonemes: phones.symbols.join(" "), audio: heard, sampleRate: config.sample_rate }, [heard.buffer]);
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
      symbols = lexicon.symbols;
      status("Model yükleniyor");
      config = await (await fetch("model.json")).json();
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
