import { useRef, useState } from "react";
import { useCameraLink } from "@/context/CameraLinkContext";
import { decodeCameraSlot } from "@/lib/camera/decodeSlot";
import { CameraLink } from "@/lib/camera/cameraLinkPlugin";

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(",")[1] ?? "");
    reader.onerror = () => reject(new Error("Failed to read the file."));
    reader.readAsDataURL(file);
  });
}

function hexPreview(base64: string, maxBytes = 32): string {
  const binary = atob(base64.slice(0, Math.ceil((maxBytes * 4) / 3) + 4));
  return [...binary]
    .slice(0, maxBytes)
    .map((c) => c.charCodeAt(0).toString(16).padStart(2, "0"))
    .join(" ");
}

/**
 * Phase 2 debug harness — not the polished Phase 4 UI. Purpose: let the
 * camera's real behavior (does connect work, do decoded values match what's
 * actually dialed into each slot on the camera's own screen) get verified
 * against real hardware before any further investment. Shows both the
 * decoded Recipe-shaped values AND the raw property dump side by side so a
 * mismatch is easy to diagnose. Connection state itself lives in
 * CameraLinkContext so it survives switching away from this tab.
 */
export function CameraDebugPage() {
  const { isNative, status, deviceName, error, slots, isScanning, connect, disconnect, scanSlots, getDeviceInfo, clearError } =
    useCameraLink();
  const [deviceInfo, setDeviceInfo] = useState<string | null>(null);
  const [isFetchingInfo, setIsFetchingInfo] = useState(false);

  // Phase 2 hard-gate harness: RAF upload + profile read, kept independent
  // of AppStateContext on purpose — this needs to validate before any
  // Preview-tab plumbing exists. Not routed through CameraLinkContext either
  // (that's Phase 5 scope) — calls the CameraLink plugin directly.
  const rawFileInputRef = useRef<HTMLInputElement>(null);
  const [rawFile, setRawFile] = useState<File | null>(null);
  const [isUploadingRaf, setIsUploadingRaf] = useState(false);
  const [rawUploadResult, setRawUploadResult] = useState<string | null>(null);
  const [isReadingProfile, setIsReadingProfile] = useState(false);
  const [rawProfileResult, setRawProfileResult] = useState<string | null>(null);

  // Phase 2.5: browse + read a RAF directly off the camera's own storage —
  // no computer/AirDrop step required. Feeds into the same upload path as
  // the manual file picker above, just with a different byte source.
  const [cameraFiles, setCameraFiles] = useState<{ index: number; name: string; size: number; date?: string }[] | null>(
    null,
  );
  const [isBrowsing, setIsBrowsing] = useState(false);
  const [isReadingCameraFile, setIsReadingCameraFile] = useState(false);

  async function uploadBase64(base64: string, label: string, size: number) {
    setIsUploadingRaf(true);
    setRawUploadResult(null);
    const startedAt = performance.now();
    try {
      const result = await CameraLink.uploadRaf({ data: base64 });
      const seconds = ((performance.now() - startedAt) / 1000).toFixed(1);
      setRawUploadResult(`ok=${result.ok} — ${label} (${size} bytes) uploaded in ${seconds}s`);
    } catch (err) {
      setRawUploadResult(err instanceof Error ? `Error: ${err.message}` : "Upload failed.");
    } finally {
      setIsUploadingRaf(false);
    }
  }

  async function handleUploadRaf() {
    if (!rawFile) return;
    const base64 = await fileToBase64(rawFile);
    await uploadBase64(base64, rawFile.name, rawFile.size);
  }

  async function handleBrowseCamera() {
    setIsBrowsing(true);
    setCameraFiles(null);
    try {
      const result = await CameraLink.listCameraFiles();
      setCameraFiles(result.files);
    } catch (err) {
      setRawUploadResult(err instanceof Error ? `Error: ${err.message}` : "Browse failed.");
    } finally {
      setIsBrowsing(false);
    }
  }

  async function handleReadAndUploadCameraFile(index: number, name: string, size: number) {
    setIsReadingCameraFile(true);
    try {
      const result = await CameraLink.readCameraFile({ index });
      await uploadBase64(result.data, name, size);
    } catch (err) {
      setRawUploadResult(err instanceof Error ? `Error: ${err.message}` : "Read from camera failed.");
    } finally {
      setIsReadingCameraFile(false);
    }
  }

  async function handleReadProfile() {
    setIsReadingProfile(true);
    setRawProfileResult(null);
    try {
      const result = await CameraLink.getRawProfile();
      setRawProfileResult(`length=${result.length} bytes\nfirst 32 bytes: ${hexPreview(result.profile)}`);
    } catch (err) {
      setRawProfileResult(err instanceof Error ? `Error: ${err.message}` : "Read profile failed.");
    } finally {
      setIsReadingProfile(false);
    }
  }

  async function handleGetDeviceInfo() {
    setIsFetchingInfo(true);
    clearError();
    setDeviceInfo(null);
    try {
      const result = await getDeviceInfo();
      setDeviceInfo(`${result.model}\n${result.raw}`);
    } catch (err) {
      setDeviceInfo(null);
      // getDeviceInfo throws directly (not routed through context's error state) — show inline instead.
      setDeviceInfo(err instanceof Error ? `Error: ${err.message}` : "Failed to get device info.");
    } finally {
      setIsFetchingInfo(false);
    }
  }

  return (
    <div className="h-full w-full overflow-y-auto bg-ink-950 p-4 text-ink-50">
      <h1 className="mb-1 text-[11px] font-bold uppercase tracking-[0.15em] text-gold-400">Camera Link — Debug</h1>
      <p className="mb-6 text-2xl font-black uppercase tracking-tight text-ink-50">
        Phase 2<span className="text-gold-400">.</span>
      </p>

      {!isNative && (
        <p className="rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-xs text-ink-400">
          Camera Link only works in the native iOS app (needs ImageCaptureCore, not available in a browser). You're
          viewing this in a regular browser/PWA context.
        </p>
      )}

      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <span
            className={`h-2 w-2 rounded-full ${
              status === "connected" ? "bg-green-400" : status === "error" ? "bg-red-400" : "bg-ink-600"
            }`}
          />
          <span className="text-xs font-bold uppercase tracking-wide text-ink-400">
            {status === "connected" ? `Connected — ${deviceName}` : status}
          </span>
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={connect}
            disabled={!isNative || status === "connecting" || status === "connected"}
            className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Connect
          </button>
          <button
            type="button"
            onClick={disconnect}
            disabled={status !== "connected"}
            className="rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Disconnect
          </button>
          <button
            type="button"
            onClick={handleGetDeviceInfo}
            disabled={status !== "connected" || isFetchingInfo}
            className="rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isFetchingInfo ? "Checking…" : "Get Device Info"}
          </button>
          <button
            type="button"
            onClick={scanSlots}
            disabled={status !== "connected" || isScanning}
            className="rounded-md border border-gold-600 bg-gold-500/10 px-4 py-2 text-xs font-bold uppercase tracking-wide text-gold-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isScanning ? "Scanning…" : "Scan 7 Slots"}
          </button>
        </div>

        {deviceInfo && (
          <pre className="whitespace-pre-wrap break-all rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-[11px] text-ink-300">
            {deviceInfo}
          </pre>
        )}

        <div className="space-y-2 rounded-md border border-gold-600/40 bg-gold-500/5 p-3">
          <p className="text-[10px] font-bold uppercase tracking-wide text-gold-400">
            Phase 2.5 — Browse camera storage (no computer/AirDrop needed)
          </p>
          <button
            type="button"
            onClick={handleBrowseCamera}
            disabled={status !== "connected" || isBrowsing}
            className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isBrowsing ? "Browsing…" : "Browse Camera"}
          </button>
          {cameraFiles && (
            <div className="space-y-1.5">
              {cameraFiles.length === 0 && <p className="text-[11px] text-ink-500">No .RAF files found on the camera.</p>}
              {cameraFiles.map((file) => (
                <div
                  key={file.index}
                  className="flex items-center justify-between gap-2 rounded-md border border-ink-800 bg-ink-900 px-3 py-2"
                >
                  <div className="min-w-0">
                    <p className="truncate text-xs font-bold text-ink-100">{file.name}</p>
                    <p className="text-[10px] text-ink-500">
                      {file.size.toLocaleString()} bytes{file.date ? ` — ${file.date}` : ""}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleReadAndUploadCameraFile(file.index, file.name, file.size)}
                    disabled={isReadingCameraFile || isUploadingRaf}
                    className="shrink-0 rounded-md border border-gold-600 bg-gold-500/10 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-gold-400 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {isReadingCameraFile ? "Reading…" : "Read + Upload"}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-2 rounded-md border border-gold-600/40 bg-gold-500/5 p-3">
          <p className="text-[10px] font-bold uppercase tracking-wide text-gold-400">
            RAW Conversion — Phase 2 hard gate (RAF upload + profile read)
          </p>
          <input
            ref={rawFileInputRef}
            type="file"
            accept=".raf"
            className="hidden"
            onChange={(event) => setRawFile(event.target.files?.[0] ?? null)}
          />
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => rawFileInputRef.current?.click()}
              className="rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300"
            >
              {rawFile ? rawFile.name : "Choose .RAF"}
            </button>
            <button
              type="button"
              onClick={handleUploadRaf}
              disabled={status !== "connected" || !rawFile || isUploadingRaf}
              className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isUploadingRaf ? "Uploading…" : "Upload RAF"}
            </button>
            <button
              type="button"
              onClick={handleReadProfile}
              disabled={status !== "connected" || isReadingProfile}
              className="rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isReadingProfile ? "Reading…" : "Read Profile (0xD185)"}
            </button>
          </div>
          {rawUploadResult && (
            <pre className="whitespace-pre-wrap break-all rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-[11px] text-ink-300">
              {rawUploadResult}
            </pre>
          )}
          {rawProfileResult && (
            <pre className="whitespace-pre-wrap break-all rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-[11px] text-ink-300">
              {rawProfileResult}
            </pre>
          )}
        </div>

        {error && (
          <pre className="whitespace-pre-wrap break-all rounded-md bg-red-500/10 px-3 py-2 text-[11px] text-red-400">
            {error}
          </pre>
        )}

        {slots && (
          <div className="space-y-3 pt-2">
            {slots.map((slot) => {
              const decoded = decodeCameraSlot(slot);
              return (
                <details key={slot.slot} className="rounded-md border border-ink-800 bg-ink-900 p-3" open>
                  <summary className="cursor-pointer text-sm font-bold text-ink-50">
                    C{slot.slot} — {slot.name}
                  </summary>
                  <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[11px] text-ink-300">
                    <span>Film Sim</span>
                    <span className="text-gold-300">{decoded.baseFilmSimulation}</span>
                    <span>Dynamic Range</span>
                    <span className="text-gold-300">{decoded.dynamicRange}</span>
                    <span>White Balance</span>
                    <span className="text-gold-300">
                      {decoded.whiteBalance.mode}
                      {decoded.whiteBalance.kelvin ? ` (${decoded.whiteBalance.kelvin}K)` : ""}
                    </span>
                    <span>WB Shift R/B</span>
                    <span className="text-gold-300">
                      {decoded.whiteBalance.shift.red} / {decoded.whiteBalance.shift.blue}
                    </span>
                    <span>Highlight / Shadow</span>
                    <span className="text-gold-300">
                      {decoded.highlightTone} / {decoded.shadowTone}
                    </span>
                    <span>Color / Sharpness</span>
                    <span className="text-gold-300">
                      {decoded.color} / {decoded.sharpness}
                    </span>
                    <span>Color Chrome / FX Blue</span>
                    <span className="text-gold-300">
                      {decoded.colorChromeEffect} / {decoded.colorChromeFxBlue}
                    </span>
                    <span>Grain</span>
                    <span className="text-gold-300">
                      {decoded.grainEffect}
                      {decoded.grainSize ? ` (${decoded.grainSize})` : ""}
                    </span>
                    <span>Clarity</span>
                    <span className="text-gold-300">{decoded.clarity}</span>
                  </div>
                  <details className="mt-3">
                    <summary className="cursor-pointer text-[10px] uppercase tracking-wide text-ink-500">
                      Raw properties
                    </summary>
                    <pre className="mt-2 overflow-x-auto text-[10px] text-ink-500">
                      {JSON.stringify(slot.properties, null, 2)}
                    </pre>
                  </details>
                </details>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
