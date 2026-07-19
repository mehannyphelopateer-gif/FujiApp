import { useRef, useState } from "react";
import { useCameraLink } from "@/context/CameraLinkContext";
import { decodeCameraSlot } from "@/lib/camera/decodeSlot";
import { CameraLink } from "@/lib/camera/cameraLinkPlugin";
import { NATIVE_IDX, patchRawProfile } from "@/lib/camera/patchRawProfile";
import { recipes } from "@/lib/recipes/loadRecipes";
import { PhotoSaver } from "@/lib/photo/photoSaverPlugin";
import { extractDetectedSettings } from "@/lib/exif/parseFujiMakerNotes";

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

function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

const BASE64_CHUNK_SIZE = 0x8000;

function uint8ArrayToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += BASE64_CHUNK_SIZE) {
    binary += String.fromCharCode(...bytes.subarray(i, i + BASE64_CHUNK_SIZE));
  }
  return btoa(binary);
}

/**
 * Compares every field this app actually patches (NATIVE_IDX) between the
 * bytes we intended to write and whatever a follow-up profile read actually
 * returned — the same read-back verification technique already proven for
 * the C1-C7 preset write path (FujiCameraSession.writePreset). Reports
 * every mismatch by field name with both values, so a wrong scale factor,
 * wrong index, or a write the camera silently rejected/reordered is
 * immediately visible instead of just "the photo looks wrong."
 */
function diffPatchedFields(intended: Uint8Array, actual: Uint8Array): string[] {
  if (intended.length !== actual.length) {
    return [`Length mismatch: intended ${intended.length} bytes, read back ${actual.length} bytes.`];
  }
  const intendedView = new DataView(intended.buffer, intended.byteOffset, intended.byteLength);
  const actualView = new DataView(actual.buffer, actual.byteOffset, actual.byteLength);
  const numParams = intendedView.getUint16(0, true);
  const off = intended.length - numParams * 4;

  const diffs: string[] = [];
  for (const [name, idx] of Object.entries(NATIVE_IDX)) {
    const expected = intendedView.getInt32(off + idx * 4, true);
    const got = actualView.getInt32(off + idx * 4, true);
    if (expected !== got) {
      diffs.push(`${name} (idx ${idx}): wrote ${expected}, read back ${got}`);
    }
  }
  return diffs;
}

/** ~1s between polls, capped at 45 tries — matches the plan's Phase 3 Go/No-Go window. */
const POLL_INTERVAL_MS = 1000;
const POLL_MAX_ATTEMPTS = 45;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
  const [cameraFiles, setCameraFiles] = useState<{ handle: number; name: string; size: number }[] | null>(null);
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
    // iOS's file picker doesn't always strictly enforce accept=".raf"
    // depending on which source (Recents/Photos/Browse) is used — a RAW+JPEG
    // pair shares the same base filename, so it's an easy mix-up to hand this
    // the JPEG twin instead. That "succeeds" here (SendObjectInfo/SendObject2
    // don't validate content, only that bytes transferred), then fails
    // opaquely several steps later when the camera has no real RAW data to
    // convert — catch it here instead, where it's immediately actionable.
    if (!rawFile.name.toLowerCase().endsWith(".raf")) {
      setRawUploadResult(`"${rawFile.name}" isn't a .RAF file — pick its RAW counterpart instead, not the JPEG.`);
      return;
    }
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

  async function handleReadAndUploadCameraFile(handle: number, name: string, size: number) {
    setIsReadingCameraFile(true);
    try {
      const result = await CameraLink.readCameraFile({ handle });
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

  // Phase 3: patch the profile with a chosen recipe, trigger conversion,
  // poll for the resulting object, download it, save it, clean up. Split
  // into separate buttons on purpose (matching the plan) so a failure at
  // any one step is immediately identifiable rather than buried in one
  // opaque "convert" call.
  const [selectedRecipeId, setSelectedRecipeId] = useState(recipes[0]?.id ?? "");
  const [conversionLog, setConversionLog] = useState<string[]>([]);
  const [isPatching, setIsPatching] = useState(false);
  const [isTriggering, setIsTriggering] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [isFinishing, setIsFinishing] = useState(false);
  const [baselineHandles, setBaselineHandles] = useState<number[] | null>(null);
  const [newHandle, setNewHandle] = useState<number | null>(null);
  // Off by default (the profile's WhiteBalance field is a documented sentinel
  // meaning "use the original shot's as-shot WB" — forcing it is unconfirmed
  // encoding territory). Exposed as a toggle rather than silently flipped, so
  // a real hardware test can directly compare "leave it" vs. "force it"
  // instead of guessing blind.
  const [forceWhiteBalance, setForceWhiteBalance] = useState(false);

  function log(line: string) {
    setConversionLog((prev) => [...prev, line]);
  }

  async function handlePatchAndWrite() {
    const recipe = recipes.find((r) => r.id === selectedRecipeId);
    if (!recipe) return;
    setIsPatching(true);
    try {
      log(`Reading current profile…`);
      const { profile, length } = await CameraLink.getRawProfile();
      log(`Read ${length} bytes. Patching for "${recipe.name}"${forceWhiteBalance ? " (forcing WB)" : ""}…`);
      const patched = patchRawProfile(base64ToUint8Array(profile), recipe, { forceWhiteBalance });
      await CameraLink.setRawProfile({ profile: uint8ArrayToBase64(patched) });
      log(`Wrote patched profile for "${recipe.name}". Reading back to verify…`);

      const readBack = await CameraLink.getRawProfile();
      const diffs = diffPatchedFields(patched, base64ToUint8Array(readBack.profile));
      if (diffs.length === 0) {
        log("Verified: every patched field matches what was written.");
      } else {
        log(`Verify MISMATCH — ${diffs.length} field(s) differ from what was written:`);
        diffs.forEach((d) => log(`  ${d}`));
      }
    } catch (err) {
      log(err instanceof Error ? `Error: ${err.message}` : "Patch + write failed.");
    } finally {
      setIsPatching(false);
    }
  }

  async function handleStartConversion() {
    setIsTriggering(true);
    setNewHandle(null);
    try {
      const { handles } = await CameraLink.listObjectHandles();
      setBaselineHandles(handles);
      log(`Baseline: ${handles.length} object(s) on camera. Triggering conversion…`);
      await CameraLink.startRawConversion();
      log("Conversion triggered.");
    } catch (err) {
      log(err instanceof Error ? `Error: ${err.message}` : "Start conversion failed.");
    } finally {
      setIsTriggering(false);
    }
  }

  async function handlePollForHandle() {
    if (!baselineHandles) return;
    setIsPolling(true);
    try {
      for (let attempt = 0; attempt < POLL_MAX_ATTEMPTS; attempt++) {
        const { handles } = await CameraLink.listObjectHandles();
        const fresh = handles.filter((h) => !baselineHandles.includes(h));
        if (fresh.length > 0) {
          setNewHandle(fresh[0]);
          log(`New object handle ${fresh[0]} appeared after ${attempt + 1}s.`);
          return;
        }
        await sleep(POLL_INTERVAL_MS);
      }
      log(`Timed out after ${POLL_MAX_ATTEMPTS}s waiting for a new object.`);
    } catch (err) {
      log(err instanceof Error ? `Error: ${err.message}` : "Poll failed.");
    } finally {
      setIsPolling(false);
    }
  }

  async function handleDownloadAndSave() {
    if (newHandle === null) return;
    setIsFinishing(true);
    try {
      log(`Downloading object ${newHandle}…`);
      const { data } = await CameraLink.downloadObject({ handle: newHandle });
      log(`Downloaded ${Math.round((data.length * 3) / 4 / 1024)} KB. Saving to Photos…`);
      const result = await PhotoSaver.saveImage({ data });
      log(`Saved to Photos. Color info: ${result.colorProfile ?? "(not reported)"}`);

      // Authoritative check: read the CONVERTED JPEG's own EXIF MakerNotes
      // (the same parser used elsewhere to detect "what recipe was used" on
      // a photo) rather than relying on eyeballing the image — this reports
      // what the camera itself claims it used for this conversion.
      const convertedFile = new File([base64ToUint8Array(data)], "converted.jpg", { type: "image/jpeg" });
      const detected = await extractDetectedSettings(convertedFile);
      if (detected) {
        log(
          `Camera's own EXIF on the result: filmSim=${detected.baseFilmSimulation} wb=${detected.whiteBalance.mode} ` +
            `shift=${detected.whiteBalance.shift.red}/${detected.whiteBalance.shift.blue} highlight=${detected.highlightTone} ` +
            `shadow=${detected.shadowTone} color=${detected.color} sharpness=${detected.sharpness}`,
        );
      } else {
        log("Camera's own EXIF on the result: none found (no Fuji MakerNotes on the converted JPEG).");
      }
    } catch (err) {
      log(err instanceof Error ? `Error: ${err.message}` : "Download + save failed.");
    } finally {
      setIsFinishing(false);
    }
  }

  async function handleDeleteObject() {
    if (newHandle === null) return;
    try {
      const { ok } = await CameraLink.deleteObject({ handle: newHandle });
      log(`Delete object ${newHandle}: ok=${ok}`);
      if (ok) setNewHandle(null);
    } catch (err) {
      log(err instanceof Error ? `Error: ${err.message}` : "Delete failed.");
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
                  key={file.handle}
                  className="flex items-center justify-between gap-2 rounded-md border border-ink-800 bg-ink-900 px-3 py-2"
                >
                  <div className="min-w-0">
                    <p className="truncate text-xs font-bold text-ink-100">{file.name}</p>
                    <p className="text-[10px] text-ink-500">{file.size.toLocaleString()} bytes</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleReadAndUploadCameraFile(file.handle, file.name, file.size)}
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

        <div className="space-y-2 rounded-md border border-gold-600/40 bg-gold-500/5 p-3">
          <p className="text-[10px] font-bold uppercase tracking-wide text-gold-400">
            Phase 3 — Convert with any recipe (real camera color science)
          </p>
          <select
            value={selectedRecipeId}
            onChange={(event) => setSelectedRecipeId(event.target.value)}
            className="w-full rounded-md border border-ink-700 bg-ink-900 px-3 py-2 text-xs text-ink-100"
          >
            {recipes.map((recipe) => (
              <option key={recipe.id} value={recipe.id}>
                {recipe.name}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-[11px] text-ink-300">
            <input
              type="checkbox"
              checked={forceWhiteBalance}
              onChange={(event) => setForceWhiteBalance(event.target.checked)}
            />
            Force white balance (default leaves the original shot's as-shot WB untouched)
          </label>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={handlePatchAndWrite}
              disabled={status !== "connected" || isPatching}
              className="rounded-md bg-gold-500 px-3 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isPatching ? "Patching…" : "1. Patch + Write Profile"}
            </button>
            <button
              type="button"
              onClick={handleStartConversion}
              disabled={status !== "connected" || isTriggering}
              className="rounded-md border border-ink-700 px-3 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isTriggering ? "Triggering…" : "2. Start Conversion"}
            </button>
            <button
              type="button"
              onClick={handlePollForHandle}
              disabled={status !== "connected" || isPolling || !baselineHandles}
              className="rounded-md border border-ink-700 px-3 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isPolling ? "Polling…" : "3. Poll for New Handle"}
            </button>
            <button
              type="button"
              onClick={handleDownloadAndSave}
              disabled={status !== "connected" || isFinishing || newHandle === null}
              className="rounded-md border border-ink-700 px-3 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isFinishing ? "Working…" : "4. Download + Save"}
            </button>
            <button
              type="button"
              onClick={handleDeleteObject}
              disabled={status !== "connected" || newHandle === null}
              className="rounded-md border border-red-800 px-3 py-2 text-xs font-bold uppercase tracking-wide text-red-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              5. Delete Object
            </button>
          </div>
          {conversionLog.length > 0 && (
            <pre className="whitespace-pre-wrap break-all rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-[11px] text-ink-300">
              {conversionLog.join("\n")}
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
