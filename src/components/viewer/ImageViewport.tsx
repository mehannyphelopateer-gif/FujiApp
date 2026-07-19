import { useCallback, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Capacitor } from "@capacitor/core";
import { useAppState } from "@/context/AppStateContext";
import { useCameraLink } from "@/context/CameraLinkContext";
import { useWebGLRenderer } from "@/engine/webgl/useWebGLRenderer";
import { PhotoSaver } from "@/lib/photo/photoSaverPlugin";
import { saveToFiles } from "@/lib/photo/shareFile";

// The Web Share API's file-sharing support is how a *web* app hands an image
// to the OS's native share sheet (which is what actually offers "Save to
// Photos"/"Save Image" — no browser API can write to the camera roll
// directly). Some browsers expose share()/canShare() for URLs/text only, so
// this checks canShare() against a real (throwaway) file rather than just
// the functions' existence — browsers without file-share support return
// false here even though the functions themselves exist.
//
// Not used at all on native iOS — see handleSaveImage's doc comment.
const supportsFileShare =
  typeof navigator !== "undefined" &&
  typeof navigator.canShare === "function" &&
  navigator.canShare({ files: [new File([""], "probe.jpg", { type: "image/jpeg" })] });

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(",")[1] ?? "");
    reader.onerror = () => reject(new Error("Failed to read the exported image."));
    reader.readAsDataURL(blob);
  });
}

export function ImageViewport() {
  const { previewUrl, recipeAdjustment, selectedRecipe, selectedFile } = useAppState();
  const { isCameraRenderMode, isConverting, convertedImageUrl, conversionError } = useCameraLink();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  // No point feeding the WebGL pipeline while a real camera-converted image
  // is what's actually being shown — the canvas stays mounted (see the
  // comment on it below) but simply doesn't draw.
  const { error } = useWebGLRenderer(canvasRef, previewUrl, previewUrl && !isCameraRenderMode ? recipeAdjustment : null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  // 0 = fully showing the recipe-applied render, 100 = fully showing the
  // untouched original. Drives a clip-path on the original <img> overlaid on
  // top of the canvas, so dragging reveals more/less of one or the other.
  const [splitPercent, setSplitPercent] = useState(50);
  const draggingRef = useRef(false);

  const updateFromClientX = useCallback((clientX: number) => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const rect = wrapper.getBoundingClientRect();
    if (rect.width === 0) return;
    const percent = ((clientX - rect.left) / rect.width) * 100;
    setSplitPercent(Math.min(100, Math.max(0, percent)));
  }, []);

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    draggingRef.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    updateFromClientX(event.clientX);
  };
  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    updateFromClientX(event.clientX);
  };
  const handlePointerUp = () => {
    draggingRef.current = false;
  };

  // Camera-render mode: the converted JPEG is already a finished, real file
  // (downloaded from the camera) — no canvas re-encode needed, just export
  // what's already there. Shared by both save actions below.
  function getExportedBlob(): Promise<Blob | null> {
    if (isCameraRenderMode) {
      if (!convertedImageUrl) return Promise.resolve(null);
      return fetch(convertedImageUrl).then((res) => res.blob());
    }
    const canvas = canvasRef.current;
    if (!canvas) return Promise.resolve(null);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
  }

  function exportFilename(): string {
    const originalName = selectedFile?.name.replace(/\.[^.]+$/, "") ?? "photo";
    const recipeSlug = selectedRecipe.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
    return `${originalName}-${recipeSlug || "recipe"}.jpg`;
  }

  // Native iOS: write straight to Photos via PHPhotoLibrary instead of the
  // share sheet. navigator.share({ files }) technically "works" inside a
  // Capacitor WKWebView, but iOS only offers a "Save Image" action in the
  // share sheet for items the OS recognizes as a real photo — a bridged
  // WKWebView's share implementation doesn't get that native glue the way
  // Safari does, so the sheet falls back to generic actions (Save to
  // Files, AirDrop, etc.) with no Photos option at all (reported directly
  // against this app's build) — hence the separate dedicated button below.
  const handleSaveToPhotos = () => {
    setSaveStatus(null);
    void (async () => {
      const blob = await getExportedBlob();
      if (!blob) return;
      try {
        const base64 = await blobToBase64(blob);
        await PhotoSaver.saveImage({ data: base64 });
        setSaveStatus("Saved to Photos.");
      } catch (err) {
        setSaveStatus(err instanceof Error ? err.message : "Failed to save the image.");
      }
    })();
  };

  const handleSaveToFiles = () => {
    setSaveStatus(null);
    void (async () => {
      const blob = await getExportedBlob();
      if (!blob) return;
      try {
        await saveToFiles(blob, exportFilename());
      } catch (err) {
        setSaveStatus(err instanceof Error ? err.message : "Failed to save the image.");
      }
    })();
  };

  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-3">
      {!previewUrl && (
        <div className="text-center text-ink-600">
          <p className="text-sm">Upload a JPEG to preview it here</p>
        </div>
      )}

      {/* The <canvas> stays mounted at all times (just hidden when there's no
          image) rather than being conditionally rendered. useWebGLRenderer's
          setup effect depends on the canvasRef object, which — being a ref —
          is stable across renders and never re-triggers the effect on its
          own when `.current` changes; if the canvas only mounted once
          previewUrl became truthy, the WebGL program would never actually
          get initialized. */}
      <div
        ref={wrapperRef}
        className={`relative min-h-0 w-full flex-1 touch-none ${previewUrl ? "" : "hidden"}`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      >
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 m-auto block max-h-full max-w-full rounded-lg object-contain shadow-2xl ${
            isCameraRenderMode ? "hidden" : ""
          }`}
        />

        {isCameraRenderMode && convertedImageUrl && (
          <img
            src={convertedImageUrl}
            alt={`${selectedRecipe.name}, rendered by the camera`}
            className="absolute inset-0 m-auto block max-h-full max-w-full rounded-lg object-contain shadow-2xl"
          />
        )}

        {isCameraRenderMode && (isConverting || conversionError) && (
          <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-black/70 p-4 text-center">
            <p className={`text-sm font-bold ${conversionError ? "text-red-400" : "text-ink-100"}`}>
              {conversionError ?? "Converting with the camera…"}
            </p>
          </div>
        )}

        {previewUrl && (
          <img
            src={previewUrl}
            alt="Original, before the recipe was applied"
            className="pointer-events-none absolute inset-0 m-auto block max-h-full max-w-full rounded-lg object-contain"
            style={{ clipPath: `inset(0 ${100 - splitPercent}% 0 0)` }}
          />
        )}

        {previewUrl && (
          <div
            className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/90"
            style={{ left: `${splitPercent}%` }}
          >
            <div className="absolute top-1/2 left-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white bg-ink-900/80 text-xs text-white">
              ⇔
            </div>
          </div>
        )}
      </div>

      {previewUrl && (
        <div className="flex shrink-0 flex-wrap items-center justify-center gap-2">
          <div className="flex rounded-md border border-ink-700 bg-ink-900 p-1">
            <button
              type="button"
              onClick={() => setSplitPercent(100)}
              className={`rounded px-3.5 py-1.5 text-xs font-bold uppercase tracking-wide transition-all ${
                splitPercent === 100 ? "bg-gold-500 text-ink-950" : "text-ink-400 hover:text-ink-100"
              }`}
            >
              Original
            </button>
            <button
              type="button"
              onClick={() => setSplitPercent(0)}
              className={`rounded px-3.5 py-1.5 text-xs font-bold uppercase tracking-wide transition-all ${
                splitPercent === 0 ? "bg-gold-500 text-ink-950" : "text-ink-400 hover:text-ink-100"
              }`}
            >
              Edited
            </button>
          </div>

          {Capacitor.isNativePlatform() && (
            <button
              type="button"
              onClick={handleSaveToPhotos}
              disabled={isCameraRenderMode && !convertedImageUrl}
              className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 transition-colors hover:bg-gold-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Save to Photos
            </button>
          )}
          <button
            type="button"
            onClick={handleSaveToFiles}
            disabled={isCameraRenderMode && !convertedImageUrl}
            className="rounded-md border border-ink-700 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-300 transition-colors hover:border-ink-500 hover:text-ink-100 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {Capacitor.isNativePlatform() || supportsFileShare ? "Save to Files" : "Download Image"}
          </button>
        </div>
      )}

      {saveStatus && <p className="rounded-lg bg-ink-900 px-3 py-2 text-xs text-ink-300">{saveStatus}</p>}
      {error && <p className="rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-400">{error}</p>}
    </div>
  );
}
