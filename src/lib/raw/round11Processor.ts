/** Local experimental X100VI RAF -> frozen Round 11 Provia/output-DR100.
 * No recipe shader, inverse LUT, gray-world AWB, exposure boost or JPEG input.
 */
import type { InferenceSession } from "onnxruntime-web";
import modelUrl from "../../../training/exports/round11/round11-seed42.onnx?url";
// Runtime 1.31.1's WebGPU bundle uses the C++/Asyncify backend. Its
// matching loader is essential; JSEP lacks the required webgpuInit API.
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import wasmModuleUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import { extractProcessorCaptureMetadata, type ProcessorCaptureMetadata } from "@/lib/exif/parseFujiMakerNotes";
import { decodeProcessorLinearRaf, extractRafPreviewJpeg, isRafFile } from "@/lib/raw/rawService";

export const ROUND11_MODEL_SHA256 = "5f155bbf83a1e112756a3c4a96554a9f87053de58dd68bb54d32e6d6f0fd264d";
export type ProcessorBackend = "webgpu" | "wasm";
export interface LinearFrame { width: number; height: number; data: Float32Array }
export interface ProcessorResult extends LinearFrame {
  capture: ProcessorCaptureMetadata;
  backend: ProcessorBackend;
  filename: string;
}

let runtimePromise: Promise<typeof import("onnxruntime-web/webgpu")> | undefined;
let modelPromise: Promise<Uint8Array> | undefined;
const sessions = new Map<ProcessorBackend, Promise<InferenceSession>>();
let queue: Promise<unknown> = Promise.resolve();

function runtime() {
  runtimePromise ??= import("onnxruntime-web/webgpu").then(ort => {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = {
      wasm: new URL(wasmUrl, window.location.href).href,
      mjs: new URL(wasmModuleUrl, window.location.href).href,
    };
    return ort;
  });
  return runtimePromise;
}

function modelBytes() {
  modelPromise ??= (async () => {
    const response = await fetch(modelUrl);
    if (!response.ok) throw new Error("The local processor model could not be loaded.");
    const bytes = await response.arrayBuffer();
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const hash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
    if (hash !== ROUND11_MODEL_SHA256) throw new Error("Processor model verification failed.");
    return new Uint8Array(bytes);
  })().catch(error => { modelPromise = undefined; throw error; });
  return modelPromise;
}

function session(backend: ProcessorBackend) {
  let pending = sessions.get(backend);
  if (!pending) {
    pending = Promise.all([runtime(), modelBytes()]).then(([ort, bytes]) => ort.InferenceSession.create(bytes, {
      executionProviders: backend === "webgpu" ? [{ name: "webgpu", preferredLayout: "NCHW" }] : ["wasm"],
    })).catch(error => { sessions.delete(backend); throw error; });
    sessions.set(backend, pending);
  }
  return pending;
}

/** Exact training-container conversion: uint16 interleaved RGB -> float32 NCHW. */
export async function readProcessorLinearFrame(blob: Blob): Promise<LinearFrame> {
  const buffer = await blob.arrayBuffer();
  if (buffer.byteLength < 32) throw new Error("Linear RAW input is incomplete.");
  const view = new DataView(buffer);
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  const width = view.getUint32(12, true), height = view.getUint32(16, true);
  if (magic !== "FJLRGB16" || view.getUint16(8, true) !== 1 || view.getUint16(10, true) !== 32
      || view.getUint16(20, true) !== 3 || view.getUint16(22, true) !== 16 || view.getUint32(28, true) !== 3
      || !width || !height || Math.max(width, height) > 1536
      || view.getUint32(24, true) !== width * height * 6 || buffer.byteLength !== 32 + width * height * 6) {
    throw new Error("Linear RAW input does not match the processor's validated format.");
  }
  const pixels = width * height, data = new Float32Array(pixels * 3);
  for (let i = 0; i < pixels; i++) for (let c = 0; c < 3; c++) {
    data[c * pixels + i] = view.getUint16(32 + (i * 3 + c) * 2, true) / 65535;
  }
  return { width, height, data };
}

/** Shared inference entry also used by the app-parity check; no image display rounding. */
export async function renderRound11Frame(frame: LinearFrame, captureDr: number, preferred?: ProcessorBackend): Promise<LinearFrame & { backend: ProcessorBackend }> {
  if (![100, 200, 400].includes(captureDr)) throw new Error("Missing or unsupported capture Dynamic Range; refusing to guess.");
  if (frame.data.length !== frame.width * frame.height * 3 || !frame.width || !frame.height || Math.max(frame.width, frame.height) > 1536) {
    throw new Error("Unsupported processor input dimensions.");
  }
  const ort = await runtime();
  const feeds = {
    linear_rgb: new ort.Tensor("float32", frame.data, [1, 3, frame.height, frame.width]),
    capture_dr_condition: new ort.Tensor("float32", new Float32Array([Math.log2(captureDr / 100) / 2]), [1, 1]),
  };
  const backends: ProcessorBackend[] = preferred ? [preferred] : "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
  let lastError: unknown;
  for (const backend of backends) {
    let outputs: Awaited<ReturnType<InferenceSession["run"]>> | undefined;
    try {
      const active = await session(backend);
      outputs = await active.run(feeds);
      const tensor = outputs.corrected_linear_rgb;
      if (!tensor || tensor.dims.length !== 4 || tensor.dims[0] !== 1 || tensor.dims[1] !== 3
          || tensor.dims[2] !== frame.height || tensor.dims[3] !== frame.width
          || !(tensor.data instanceof Float32Array) || !tensor.data.every(Number.isFinite)) {
        throw new Error("Processor returned invalid pixel data.");
      }
      return { width: frame.width, height: frame.height, data: new Float32Array(tensor.data), backend };
    } catch (error) {
      lastError = error;
      const failed = sessions.get(backend);
      sessions.delete(backend);
      if (failed) void failed.then(s => s.release()).catch(() => {});
    } finally {
      if (outputs) for (const tensor of Object.values(outputs)) tensor.dispose();
    }
  }
  throw new Error(`Local processor could not render this image. ${lastError instanceof Error ? lastError.message : "Runtime unavailable."}`);
}

export function processRound11Raf(file: File, onProgress: (text: string) => void, signal?: AbortSignal): Promise<ProcessorResult> {
  const checkCancelled = () => { if (signal?.aborted) throw new DOMException("Cancelled", "AbortError"); };
  const work = queue.then(async () => {
    checkCancelled();
    if (!isRafFile(file)) throw new Error("Choose an original X100VI RAF file. JPEG processing is not available in this experimental mode.");
    onProgress("Checking camera settings…");
    const capture = await extractProcessorCaptureMetadata(await extractRafPreviewJpeg(file));
    checkCancelled();
    onProgress("Decoding RAW sensor data locally…");
    const decoded = await decodeProcessorLinearRaf(file);
    checkCancelled();
    const frame = await readProcessorLinearFrame(decoded.blob);
    onProgress("Rendering Provia with the new processor…");
    const rendered = await renderRound11Frame(frame, capture.captureDynamicRange);
    checkCancelled();
    return { ...rendered, capture, filename: file.name };
  });
  queue = work.catch(() => {});
  return work;
}

/** Only display/PNG encoding is quantized. RAW input and inference stay linear float32. */
export function drawProcessorResult(canvas: HTMLCanvasElement, frame: LinearFrame): void {
  canvas.width = frame.width; canvas.height = frame.height;
  const context = canvas.getContext("2d", { colorSpace: "srgb" });
  if (!context) throw new Error("Canvas display is unavailable.");
  const rgba = new Uint8ClampedArray(frame.width * frame.height * 4), pixels = frame.width * frame.height;
  for (let i = 0; i < pixels; i++) {
    for (let c = 0; c < 3; c++) {
      const linear = Math.min(1, Math.max(0, frame.data[c * pixels + i]));
      const srgb = linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055;
      rgba[i * 4 + c] = Math.round(255 * srgb);
    }
    rgba[i * 4 + 3] = 255;
  }
  context.putImageData(new ImageData(rgba, frame.width, frame.height), 0, 0);
}
