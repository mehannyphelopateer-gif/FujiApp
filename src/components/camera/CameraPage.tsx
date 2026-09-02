import { useEffect, useMemo, useRef, useState } from "react";
import { useAppState } from "@/context/AppStateContext";
import { useCameraLink, type WriteResult } from "@/context/CameraLinkContext";
import { CameraLink } from "@/lib/camera/cameraLinkPlugin";
import { RecipeGrid } from "@/components/recipes/RecipeGrid";
import { RecipeQaSweep } from "@/components/camera/RecipeQaSweep";
import { CalibrationCapture } from "@/components/camera/CalibrationCapture";
import { PhotoSaver } from "@/lib/photo/photoSaverPlugin";
import { saveToFiles } from "@/lib/photo/shareFile";
import { base64ToBlob } from "@/lib/camera/base64";
import { decodeCameraSlot } from "@/lib/camera/decodeSlot";
import { extractRafPreviewJpeg } from "@/lib/raw/rawService";
import { recipes as allRecipes } from "@/lib/recipes/loadRecipes";
import { mapCameraModelToSensorGeneration } from "@/lib/exif/sensorGenerations";

const SLOT_NUMBERS = [1, 2, 3, 4, 5, 6, 7];

interface RafCandidate {
  file: File;
  thumbnailUrl: string;
}

function base64ToRafFile(base64: string, name: string): File {
  return new File([base64ToBlob(base64)], name);
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(",")[1] ?? "");
    reader.onerror = () => reject(new Error("Failed to read the converted image."));
    reader.readAsDataURL(blob);
  });
}

/**
 * Load a .RAF straight off the camera (or from Files), pick any recipe from
 * the library, and get back a real camera-converted preview — the camera's
 * own color science, not a software approximation. See CameraLinkContext's
 * convertWithRecipe for the actual upload/patch/trigger/download pipeline;
 * this page is just the simple front end for it.
 */
export function CameraPage() {
  const { selectedRecipe, selectedRecipeId } = useAppState();
  const {
    isNative,
    status,
    deviceName,
    error,
    connect,
    clearError,
    isConverting,
    convertedImageUrl,
    conversionError,
    conversionWarning,
    convertWithRecipe,
    slots,
    isScanning,
    isWriting,
    scanSlots,
    writeRecipeToSlot,
  } = useCameraLink();

  const [rafFile, setRafFile] = useState<File | null>(null);
  const [candidates, setCandidates] = useState<RafCandidate[] | null>(null);
  const [isBuildingThumbnails, setIsBuildingThumbnails] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Browse Camera (direct PTP enumeration) is confirmed dead in USB RAW
  // CONV./BACKUP RESTORE mode (that mode doesn't expose the SD card's
  // library over PTP at all), but worth testing again whenever connected in
  // a different USB mode (e.g. USB Card Reader) — same code, might behave
  // differently since it's a completely different camera-side USB personality.
  const [cameraFiles, setCameraFiles] = useState<{ handle: number; name: string; size: number }[] | null>(null);
  // handle -> base64 JPEG thumbnail, or null once confirmed unavailable — absent key means still loading.
  const [cameraThumbnails, setCameraThumbnails] = useState<Record<number, string | null>>({});
  const [browseDiagnostic, setBrowseDiagnostic] = useState<{ totalObjectCount: number; sampleFilenames: string[] } | null>(
    null,
  );
  const [isBrowsing, setIsBrowsing] = useState(false);
  const [isLoadingCameraFile, setIsLoadingCameraFile] = useState(false);

  const [slotToWrite, setSlotToWrite] = useState<number | null>(null);
  const [writeResult, setWriteResult] = useState<WriteResult | null>(null);

  // Filtered by the actually-connected camera body's real sensor generation
  // (its PTP-reported device name), not by whatever photo happens to be
  // loaded in the Preview tab — writing/converting a recipe the connected
  // body doesn't physically support is a real, silent failure mode.
  const cameraSensorGeneration = useMemo(
    () => (status === "connected" ? mapCameraModelToSensorGeneration(deviceName) : null),
    [status, deviceName],
  );
  const cameraCompatibleRecipes = useMemo(() => {
    if (!cameraSensorGeneration) return allRecipes; // fail open: unrecognized body shows everything
    return allRecipes.filter((recipe) => recipe.compatibleSensors.includes(cameraSensorGeneration));
  }, [cameraSensorGeneration]);

  // Deliberately a guard, not an auto-reset of the shared selectedRecipeId —
  // AppStateContext already has its own reset effect for the Preview tab's
  // photo-derived compatible list, and if the two lists ever disagreed
  // (photo shot on one body, cable-connected to another), two effects each
  // "correcting" the same shared value toward a different list would fight
  // forever. This just gates what this page does with the selection instead.
  const isSelectedRecipeCompatible = cameraCompatibleRecipes.some((recipe) => recipe.id === selectedRecipeId);

  useEffect(() => {
    if (!rafFile || !isSelectedRecipeCompatible) return;
    void convertWithRecipe(selectedRecipe, rafFile);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rafFile, selectedRecipe, isSelectedRecipeCompatible]);

  // A file loaded via Browse Camera while in USB Card Reader mode can only
  // be converted after switching the camera to USB RAW CONV./BACKUP
  // RESTORE mode and reconnecting (the upload/convert commands are Fuji
  // vendor extensions only available in that mode) — reconnecting doesn't
  // change rafFile/selectedRecipe, so the effect above won't auto-retry on
  // its own. This gives an explicit way to try again once reconnected.
  function handleRetryConversion() {
    if (!rafFile) return;
    void convertWithRecipe(selectedRecipe, rafFile);
  }

  function revokeCandidates(list: RafCandidate[] | null) {
    list?.forEach((c) => URL.revokeObjectURL(c.thumbnailUrl));
  }

  async function handleBrowseCamera() {
    setIsBrowsing(true);
    setLoadError(null);
    setBrowseDiagnostic(null);
    setCameraFiles(null);
    setCameraThumbnails({});
    try {
      const result = await CameraLink.listCameraFiles();
      setCameraFiles(result.files);
      if (result.files.length === 0) {
        setBrowseDiagnostic({ totalObjectCount: result.totalObjectCount, sampleFilenames: result.sampleFilenames });
      } else {
        // Fetch thumbnails independently per file (not the whole 80MB+
        // object) so the grid fills in progressively instead of blocking on
        // the slowest/least available one.
        for (const file of result.files) {
          void CameraLink.getCameraFileThumbnail({ handle: file.handle })
            .then(({ data }) => setCameraThumbnails((prev) => ({ ...prev, [file.handle]: data })))
            .catch(() => setCameraThumbnails((prev) => ({ ...prev, [file.handle]: null })));
        }
      }
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Couldn't list files on the camera.");
    } finally {
      setIsBrowsing(false);
    }
  }

  async function handleLoadCameraFile(handle: number, name: string) {
    setIsLoadingCameraFile(true);
    setLoadError(null);
    setSaveStatus(null);
    try {
      const { data } = await CameraLink.readCameraFile({ handle });
      setRafFile(base64ToRafFile(data, name));
      setCameraFiles(null);
      setCameraThumbnails({});
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Couldn't read that file from the camera.");
    } finally {
      setIsLoadingCameraFile(false);
    }
  }

  async function handleChooseFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return;
    const files = Array.from(fileList).filter((f) => f.name.toLowerCase().endsWith(".raf"));
    if (files.length === 0) {
      setLoadError("No .RAF files in that selection — pick the RAW files, not JPEGs.");
      return;
    }
    setLoadError(null);
    setSaveStatus(null);

    // A single pick is unambiguous — skip straight to it, no grid needed.
    if (files.length === 1) {
      setRafFile(files[0]);
      return;
    }

    setIsBuildingThumbnails(true);
    const built: RafCandidate[] = [];
    for (const file of files) {
      try {
        const preview = await extractRafPreviewJpeg(file);
        built.push({ file, thumbnailUrl: URL.createObjectURL(preview) });
      } catch {
        // Skip files whose embedded preview can't be read — they just won't appear as a pickable option.
      }
    }
    setIsBuildingThumbnails(false);
    if (built.length === 0) {
      setLoadError("Couldn't read a preview from any of those files.");
      return;
    }
    setCandidates(built);
  }

  function handlePickCandidate(picked: RafCandidate) {
    revokeCandidates(candidates?.filter((c) => c !== picked) ?? null);
    setCandidates(null);
    setRafFile(picked.file);
  }

  function slotLabel(slot: number): string {
    const found = slots?.find((s) => s.slot === slot);
    if (!found) return "Unknown — scan first";
    return `${found.name || "(unnamed)"} · ${decodeCameraSlot(found).baseFilmSimulation}`;
  }

  async function handleWriteToSlot(slot: number) {
    clearError();
    setWriteResult(null);
    const result = await writeRecipeToSlot(selectedRecipe, slot);
    setWriteResult(result);
    setSlotToWrite(null);
    if (result.ok) void scanSlots();
  }

  async function handleSaveToPhotos() {
    if (!convertedImageUrl) return;
    setSaveStatus(null);
    try {
      const blob = await (await fetch(convertedImageUrl)).blob();
      const base64 = await blobToBase64(blob);
      await PhotoSaver.saveImage({ data: base64 });
      setSaveStatus("Saved to Photos.");
    } catch (err) {
      setSaveStatus(err instanceof Error ? err.message : "Failed to save the image.");
    }
  }

  async function handleSaveToFiles() {
    if (!convertedImageUrl) return;
    setSaveStatus(null);
    try {
      const blob = await (await fetch(convertedImageUrl)).blob();
      const slug = selectedRecipe.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
      await saveToFiles(blob, `${slug || "recipe"}.jpg`);
    } catch (err) {
      setSaveStatus(err instanceof Error ? err.message : "Failed to save the image.");
    }
  }

  return (
    <div className="h-full w-full overflow-y-auto bg-ink-950 p-4 text-ink-50">
      <h1 className="mb-1 text-[11px] font-bold uppercase tracking-[0.15em] text-gold-400">Render with Camera</h1>
      <p className="mb-4 text-xs text-ink-400">
        Load a RAW file, pick any recipe, and get back a preview converted by the camera's own processor.
      </p>

      {!isNative && (
        <p className="mb-4 rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-xs text-ink-400">
          This only works in the native iOS app — it needs a direct cable connection to the camera, not available in
          a browser.
        </p>
      )}

      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5">
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${
                status === "connected" ? "bg-green-400" : status === "error" ? "bg-red-400" : "bg-ink-600"
              }`}
            />
            <span className="text-xs font-bold uppercase tracking-wide text-ink-300">
              {status === "connected" ? `Connected — ${deviceName}` : status === "connecting" ? "Connecting…" : "Not connected"}
            </span>
          </div>
          {status !== "connected" && (
            <button
              type="button"
              onClick={connect}
              disabled={!isNative || status === "connecting"}
              className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Connect Camera
            </button>
          )}
        </div>

        <div className="space-y-2">
          <p className="text-[11px] font-bold uppercase tracking-wide text-ink-500">Load a RAW file (for a preview)</p>

          {rafFile ? (
            <div className="flex items-center justify-between gap-2 rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-xs">
              <span className="truncate text-ink-300">{rafFile.name}</span>
              <button
                type="button"
                onClick={() => {
                  revokeCandidates(candidates);
                  setCandidates(null);
                  setRafFile(null);
                  setSaveStatus(null);
                }}
                className="shrink-0 font-bold uppercase tracking-wide text-gold-400 hover:text-gold-300"
              >
                Change
              </button>
            </div>
          ) : (
            <>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isBuildingThumbnails}
                  className="flex-1 rounded-md bg-gold-500 px-4 py-2.5 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {isBuildingThumbnails ? "Loading previews…" : "Pick a RAW File"}
                </button>
                <button
                  type="button"
                  onClick={handleBrowseCamera}
                  disabled={status !== "connected" || isBrowsing}
                  className="flex-1 rounded-md border border-ink-700 px-4 py-2.5 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {isBrowsing ? "Browsing…" : "Browse Camera"}
                </button>
              </div>
              <p className="text-[10px] text-ink-600">
                "Pick a RAW File" opens Files — select one or more .RAF files, and with more than one you'll see the
                actual photos to pick from, not just filenames. "Browse Camera" reads the camera's storage directly
                (works in some USB modes, not others — worth trying either way).
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".raf"
                multiple
                className="hidden"
                onChange={(event) => handleChooseFiles(event.target.files)}
              />
            </>
          )}

          {candidates && !rafFile && (
            <div className="grid grid-cols-3 gap-2">
              {candidates.map((candidate) => (
                <button
                  key={candidate.file.name}
                  type="button"
                  onClick={() => handlePickCandidate(candidate)}
                  className="overflow-hidden rounded-md border border-ink-800 text-left"
                >
                  <div className="flex aspect-square items-center justify-center bg-black/30">
                    <img src={candidate.thumbnailUrl} alt={candidate.file.name} className="h-full w-full object-cover" />
                  </div>
                  <p className="truncate bg-ink-900 px-1.5 py-1 text-[10px] text-ink-300">{candidate.file.name}</p>
                </button>
              ))}
            </div>
          )}

          {cameraFiles && !rafFile && (
            <div className="space-y-1.5">
              {cameraFiles.length === 0 && (
                <div className="space-y-1 rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5">
                  <p className="text-[11px] text-ink-500">No .RAF files found on the camera.</p>
                  {browseDiagnostic && (
                    <p className="text-[10px] text-ink-600">
                      {browseDiagnostic.totalObjectCount === 0
                        ? "The camera reported 0 objects at all — the connection or camera mode is likely the issue, not file filtering."
                        : `The camera reported ${browseDiagnostic.totalObjectCount} object(s), but none ended in .raf. Examples: ${
                            browseDiagnostic.sampleFilenames.join(", ") || "(none)"
                          }`}
                    </p>
                  )}
                </div>
              )}
              {cameraFiles.length > 0 && (
                <div className="grid grid-cols-3 gap-2">
                  {cameraFiles.map((file) => {
                    const thumbnail = cameraThumbnails[file.handle];
                    return (
                      <button
                        key={file.handle}
                        type="button"
                        onClick={() => handleLoadCameraFile(file.handle, file.name)}
                        disabled={isLoadingCameraFile}
                        className="overflow-hidden rounded-md border border-ink-800 text-left disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        <div className="flex aspect-square items-center justify-center bg-black/30">
                          {thumbnail ? (
                            <img
                              src={`data:image/jpeg;base64,${thumbnail}`}
                              alt={file.name}
                              className="h-full w-full object-cover"
                            />
                          ) : thumbnail === null ? (
                            <span className="p-1 text-center text-[10px] text-ink-600">No preview</span>
                          ) : (
                            <span className="p-1 text-center text-[10px] text-ink-600">Loading…</span>
                          )}
                        </div>
                        <p className="truncate bg-ink-900 px-1.5 py-1 text-[10px] text-ink-300">{file.name}</p>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {loadError && <p className="text-[11px] text-red-400">{loadError}</p>}
        </div>

        {rafFile && (
          <div className="space-y-2">
            <p className="text-[11px] font-bold uppercase tracking-wide text-ink-500">Preview</p>
            <div className="relative flex aspect-[3/2] w-full items-center justify-center overflow-hidden rounded-md border border-ink-800 bg-black/30">
              {convertedImageUrl && (
                <img
                  src={convertedImageUrl}
                  alt={`${selectedRecipe.name}, rendered by the camera`}
                  className="h-full w-full object-contain"
                />
              )}
              {(isConverting || conversionError) && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/70 p-4 text-center">
                  <p className={`text-sm font-bold ${conversionError ? "text-red-400" : "text-ink-100"}`}>
                    {conversionError ?? "Converting with the camera…"}
                  </p>
                </div>
              )}
              {!convertedImageUrl && !isConverting && !conversionError && (
                <p className="p-4 text-center text-xs text-ink-500">Pick a recipe below to see it here.</p>
              )}
            </div>
            {conversionError && (
              <div className="space-y-1.5">
                <button
                  type="button"
                  onClick={handleRetryConversion}
                  className="w-full rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300"
                >
                  Retry Conversion
                </button>
                <p className="text-center text-[10px] text-ink-600">
                  If this file came from Browse Camera, converting needs the camera in USB RAW CONV./BACKUP RESTORE
                  mode — switch modes and reconnect, then retry.
                </p>
              </div>
            )}
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleSaveToPhotos}
                disabled={!convertedImageUrl}
                className="flex-1 rounded-md bg-gold-500 px-4 py-2.5 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Save to Photos
              </button>
              <button
                type="button"
                onClick={handleSaveToFiles}
                disabled={!convertedImageUrl}
                className="flex-1 rounded-md border border-ink-700 px-4 py-2.5 text-xs font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Save to Files
              </button>
            </div>
            {saveStatus && <p className="text-center text-xs text-ink-300">{saveStatus}</p>}
            {conversionWarning && <p className="text-center text-xs text-amber-400">{conversionWarning}</p>}
          </div>
        )}

        {rafFile && status === "connected" && (
          <details className="rounded-md border border-ink-800 bg-ink-900/50 p-3">
            <summary className="cursor-pointer text-[11px] font-bold uppercase tracking-wide text-ink-500">
              Advanced: QA Sweep All Recipes
            </summary>
            <div className="mt-3">
              <RecipeQaSweep rafFile={rafFile} recipes={cameraCompatibleRecipes} />
            </div>
          </details>
        )}

        {rafFile && status === "connected" && (
          <details className="rounded-md border border-ink-800 bg-ink-900/50 p-3">
            <summary className="cursor-pointer text-[11px] font-bold uppercase tracking-wide text-ink-500">
              Advanced: LUT Calibration Capture
            </summary>
            <div className="mt-3">
              <CalibrationCapture rafFile={rafFile} />
            </div>
          </details>
        )}

        <div className="space-y-2 rounded-md border border-ink-800 bg-ink-900/50 p-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[11px] font-bold uppercase tracking-wide text-ink-500">Custom Slots (C1-C7)</p>
            <button
              type="button"
              onClick={scanSlots}
              disabled={status !== "connected" || isScanning}
              className="shrink-0 rounded-md border border-ink-700 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-ink-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isScanning ? "Scanning…" : slots ? "Rescan" : "Scan Slots"}
            </button>
          </div>

          {slots && (
            <div className="space-y-1.5">
              {SLOT_NUMBERS.map((slot) => (
                <div key={slot} className="rounded-md border border-ink-800 bg-ink-900 px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0">
                      <span className="font-bold text-ink-50">C{slot}</span>{" "}
                      <span className="truncate text-xs text-ink-400">{slotLabel(slot)}</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setWriteResult(null);
                        setSlotToWrite(slot);
                      }}
                      disabled={isScanning || isWriting || !isSelectedRecipeCompatible}
                      title={!isSelectedRecipeCompatible ? "Not compatible with your connected camera" : undefined}
                      className="shrink-0 rounded-md border border-gold-600 bg-gold-500/10 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-gold-400 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      Write {selectedRecipe.name}
                    </button>
                  </div>

                  {slotToWrite === slot && (
                    <div className="mt-2 space-y-2 rounded-md border border-gold-700/50 bg-gold-500/5 p-2.5">
                      <p className="text-[11px] text-ink-200">
                        Overwrite <span className="font-bold text-gold-300">C{slot}</span> (currently{" "}
                        <span className="italic">{slotLabel(slot)}</span>) with{" "}
                        <span className="font-bold text-gold-300">{selectedRecipe.name}</span>? Can't be undone from
                        the app.
                      </p>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => setSlotToWrite(null)}
                          disabled={isWriting}
                          className="flex-1 rounded-md border border-ink-700 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-ink-300 disabled:opacity-40"
                        >
                          Cancel
                        </button>
                        <button
                          type="button"
                          onClick={() => handleWriteToSlot(slot)}
                          disabled={isWriting}
                          className="flex-1 rounded-md bg-gold-500 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
                        >
                          {isWriting ? "Writing…" : "Confirm"}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {writeResult && (
            <p className={`text-[11px] font-bold ${writeResult.ok ? "text-green-400" : "text-red-400"}`}>
              {writeResult.ok ? "Written." : "Write failed."}
              {writeResult.warnings.length > 0 && ` ${writeResult.warnings.join(" ")}`}
            </p>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[11px] font-bold uppercase tracking-wide text-ink-500">Pick a recipe</p>
            {status === "connected" && cameraSensorGeneration && (
              <p className="text-[10px] text-ink-500">Showing recipes for {cameraSensorGeneration}</p>
            )}
          </div>
          {status === "connected" && !isSelectedRecipeCompatible && (
            <p className="text-[11px] text-red-400">
              "{selectedRecipe.name}" isn't compatible with your connected {deviceName} — pick one below.
            </p>
          )}
          <RecipeGrid recipes={status === "connected" ? cameraCompatibleRecipes : undefined} />
        </div>

        {error && (
          <div className="flex items-start justify-between gap-2 rounded-md bg-red-500/10 px-3 py-2">
            <p className="text-[11px] text-red-400">{error}</p>
            <button type="button" onClick={clearError} className="shrink-0 text-[11px] font-bold text-red-300">
              Dismiss
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
