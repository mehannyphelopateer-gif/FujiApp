import { useEffect, useRef, useState } from "react";
import { processRound11Raf, drawProcessorResult, type ProcessorResult } from "@/lib/raw/round11Processor";
import { saveToFiles } from "@/lib/photo/shareFile";

export default function ExperimentalRawPreview() {
  const inputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const [result, setResult] = useState<ProcessorResult | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => () => { controllerRef.current?.abort(); }, []);
  useEffect(() => {
    if (result && canvasRef.current) {
      try { drawProcessorResult(canvasRef.current, result); }
      catch (e) { setError(e instanceof Error ? e.message : "Preview display failed."); }
    }
  }, [result]);

  async function upload(file?: File) {
    if (!file || busy) return;
    controllerRef.current?.abort();
    const controller = new AbortController(); controllerRef.current = controller;
    setResult(null); setError(null); setBusy(true); setStatus("Preparing RAW file…");
    try {
      const next = await processRound11Raf(file, text => { if (!controller.signal.aborted) setStatus(text); }, controller.signal);
      if (!controller.signal.aborted) { setResult(next); setStatus("Rendered locally. No camera connection needed."); }
    } catch (e) {
      if (!controller.signal.aborted) { setError(e instanceof Error ? e.message : "RAW processing failed."); setStatus(""); }
    } finally { if (!controller.signal.aborted) setBusy(false); }
  }

  async function save() {
    if (!result || !canvasRef.current || saving) return;
    setSaving(true); setError(null);
    try {
      const blob = await new Promise<Blob>((resolve, reject) => canvasRef.current!.toBlob(value => value ? resolve(value) : reject(new Error("PNG export failed.")), "image/png"));
      await saveToFiles(blob, `${result.filename.replace(/\.raf$/i, "")}-experimental-provia-dr100.png`);
    } catch (e) { setError(e instanceof Error ? e.message : "Saving failed."); }
    finally { setSaving(false); }
  }

  return (
    <main className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto p-4 lg:p-6">
      <div>
        <h1 className="text-lg font-bold text-ink-50">Experimental RAW processor</h1>
        <p className="mt-1 text-sm text-ink-300">X100VI RAF → Provia · DR100 output · As-shot white balance</p>
        <p className="mt-2 text-xs text-ink-400">Local test build · 1536px preview · Other recipes and JPEG input are not available in this mode. Fuji-perfect matching is not guaranteed.</p>
      </div>
      <input ref={inputRef} type="file" accept=".raf" className="hidden" aria-label="Upload RAF for experimental processor"
        disabled={busy || saving} onChange={e => { const file = e.target.files?.[0]; e.target.value = ""; void upload(file); }} />
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" disabled={busy || saving} onClick={() => inputRef.current?.click()}
          className="rounded-md bg-gold-500 px-4 py-2 text-sm font-bold text-ink-950 disabled:opacity-50">
          {busy ? "Processing RAF…" : result ? "Choose another RAF" : "Upload X100VI RAF"}
        </button>
        {result && <button type="button" disabled={saving || busy} onClick={() => void save()}
          className="rounded-md border border-ink-600 px-4 py-2 text-sm text-ink-100 disabled:opacity-50">{saving ? "Saving…" : "Save preview PNG"}</button>}
        <span role="status" className="text-xs text-ink-300">{status}</span>
      </div>
      {error && <p role="alert" className="rounded-md bg-red-500/10 p-3 text-sm text-red-400">{error}</p>}
      {result && <p className="text-xs text-ink-400">{result.filename} · Capture DR{result.capture.captureDynamicRange} → Output DR100 · {result.width} × {result.height} · {result.backend === "webgpu" ? "WebGPU" : "WASM"}</p>}
      <div className="flex min-h-[40vh] flex-1 items-center justify-center rounded-lg bg-black/30 p-2">
        <canvas ref={canvasRef} aria-label="Experimental Provia rendering" className={`max-h-full max-w-full rounded-lg object-contain ${result ? "" : "hidden"}`} />
        {!result && <p className="text-center text-sm text-ink-500">{busy ? "Large RAF files can take a minute to decode." : "Choose a RAF to test the new processor."}</p>}
      </div>
      <p className="text-xs text-ink-500">PNG exports are 8-bit display previews at the working resolution.</p>
    </main>
  );
}
