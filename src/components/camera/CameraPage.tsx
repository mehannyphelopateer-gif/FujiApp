import { useEffect, useRef, useState } from "react";
import { useAppState } from "@/context/AppStateContext";
import { useCameraLink } from "@/context/CameraLinkContext";
import { CameraLink } from "@/lib/camera/cameraLinkPlugin";
import { RecipeGrid } from "@/components/recipes/RecipeGrid";
import { PhotoSaver } from "@/lib/photo/photoSaverPlugin";
import { base64ToBlob } from "@/lib/camera/base64";

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
  const { selectedRecipe } = useAppState();
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
  } = useCameraLink();

  const [rafFile, setRafFile] = useState<File | null>(null);
  const [cameraFiles, setCameraFiles] = useState<{ handle: number; name: string; size: number }[] | null>(null);
  const [isBrowsing, setIsBrowsing] = useState(false);
  const [isLoadingRaf, setIsLoadingRaf] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!rafFile) return;
    void convertWithRecipe(selectedRecipe, rafFile);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rafFile, selectedRecipe]);

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

        <div className="space-y-2">
          <p className="text-[11px] font-bold uppercase tracking-wide text-ink-500">1. Load a RAW file</p>

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
            <p className="text-[11px] font-bold uppercase tracking-wide text-ink-500">2. Pick a recipe</p>
            <RecipeGrid />
          </div>
        )}

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
