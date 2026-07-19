import { useEffect, useMemo, useRef, useState } from "react";
import { useAppState } from "@/context/AppStateContext";
import { useCameraLink, type WriteResult } from "@/context/CameraLinkContext";
import { CameraLink } from "@/lib/camera/cameraLinkPlugin";
import { RecipeGrid } from "@/components/recipes/RecipeGrid";
import { RecipeQaSweep } from "@/components/camera/RecipeQaSweep";
import { PhotoSaver } from "@/lib/photo/photoSaverPlugin";
import { base64ToBlob } from "@/lib/camera/base64";
import { decodeCameraSlot } from "@/lib/camera/decodeSlot";
import { recipes as allRecipes } from "@/lib/recipes/loadRecipes";
import { mapCameraModelToSensorGeneration } from "@/lib/exif/sensorGenerations";

const SLOT_NUMBERS = [1, 2, 3, 4, 5, 6, 7];

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
    convertWithRecipe,
    slots,
    isScanning,
    isWriting,
    scanSlots,
    writeRecipeToSlot,
  } = useCameraLink();

  const [rafFile, setRafFile] = useState<File | null>(null);
  const [cameraFiles, setCameraFiles] = useState<{ handle: number; name: string; size: number }[] | null>(null);
  const [isBrowsing, setIsBrowsing] = useState(false);
  const [isLoadingRaf, setIsLoadingRaf] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  async function handleBrowseCamera() {
    setIsBrowsing(true);
    setLoadError(null);
    try {
      const result = await CameraLink.listCameraFiles();
      setCameraFiles(result.files);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Couldn't list files on the camera.");
    } finally {
      setIsBrowsing(false);
    }
  }

  async function handleLoadCameraFile(handle: number, name: string) {
    setIsLoadingRaf(true);
    setLoadError(null);
    setSaveStatus(null);
    try {
      const { data } = await CameraLink.readCameraFile({ handle });
      setRafFile(base64ToRafFile(data, name));
      setCameraFiles(null);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Couldn't read that file from the camera.");
    } finally {
      setIsLoadingRaf(false);
    }
  }

  function handleChooseFile(file: File | null) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".raf")) {
      setLoadError(`"${file.name}" isn't a .RAF file — pick its RAW counterpart instead.`);
      return;
    }
    setLoadError(null);
    setSaveStatus(null);
    setCameraFiles(null);
    setRafFile(file);
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

  async function handleSaveImage() {
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

        <div className="space-y-2">
          <p className="text-[11px] font-bold uppercase tracking-wide text-ink-500">Load a RAW file (for a preview)</p>

          {rafFile ? (
            <div className="flex items-center justify-between gap-2 rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-xs">
              <span className="truncate text-ink-300">{rafFile.name}</span>
              <button
                type="button"
                onClick={() => {
                  setRafFile(null);
                  setSaveStatus(null);
                }}
                className="shrink-0 font-bold uppercase tracking-wide text-gold-400 hover:text-gold-300"
              >
                Change
              </button>
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={handleBrowseCamera}
                disabled={status !== "connected" || isBrowsing}
                className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {isBrowsing ? "Browsing…" : "Browse Camera"}
              </button>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300"
              >
                Choose File
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".raf"
                className="hidden"
                onChange={(event) => handleChooseFile(event.target.files?.[0] ?? null)}
              />
            </div>
          )}

          {cameraFiles && !rafFile && (
            <div className="space-y-1.5">
              {cameraFiles.length === 0 && (
                <p className="text-[11px] text-ink-500">No .RAF files found on the camera.</p>
              )}
              {cameraFiles.map((file) => (
                <button
                  key={file.handle}
                  type="button"
                  onClick={() => handleLoadCameraFile(file.handle, file.name)}
                  disabled={isLoadingRaf}
                  className="flex w-full items-center justify-between gap-2 rounded-md border border-ink-800 bg-ink-900 px-3 py-2 text-left disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <span className="truncate text-xs font-bold text-ink-100">{file.name}</span>
                  <span className="shrink-0 text-[10px] text-ink-500">
                    {isLoadingRaf ? "Loading…" : `${Math.round(file.size / 1024 / 1024)} MB`}
                  </span>
                </button>
              ))}
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
                <p className="p-4 text-center text-xs text-ink-500">Pick a recipe above to see it here.</p>
              )}
            </div>
            <button
              type="button"
              onClick={handleSaveImage}
              disabled={!convertedImageUrl}
              className="w-full rounded-md bg-gold-500 px-4 py-2.5 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Save to Photos
            </button>
            {saveStatus && <p className="text-center text-xs text-ink-300">{saveStatus}</p>}
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
