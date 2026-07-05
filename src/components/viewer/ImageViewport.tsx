import { useCallback, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { useAppState } from "@/context/AppStateContext";
import { useWebGLRenderer } from "@/engine/webgl/useWebGLRenderer";

// The Web Share API's file-sharing support is how a web app hands an image
// to the OS's native share sheet (which is what actually offers "Save to
// Photos"/"Save Image" — no browser API can write to the camera roll
// directly). Some browsers expose share()/canShare() for URLs/text only, so
// this checks canShare() against a real (throwaway) file rather than just
// the functions' existence — browsers without file-share support return
// false here even though the functions themselves exist.
const supportsFileShare =
  typeof navigator !== "undefined" &&
  typeof navigator.canShare === "function" &&
  navigator.canShare({ files: [new File([""], "probe.jpg", { type: "image/jpeg" })] });

export function ImageViewport() {
  const { previewUrl, recipeAdjustment, selectedRecipe, selectedFile } = useAppState();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const { error } = useWebGLRenderer(canvasRef, previewUrl, previewUrl ? recipeAdjustment : null);

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

  const handleSaveImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob(
      async (blob) => {
        if (!blob) return;
        const originalName = selectedFile?.name.replace(/\.[^.]+$/, "") ?? "photo";
        const recipeSlug = selectedRecipe.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
        const filename = `${originalName}-${recipeSlug || "recipe"}.jpg`;
        const file = new File([blob], filename, { type: "image/jpeg" });

        if (supportsFileShare && navigator.canShare({ files: [file] })) {
          try {
            await navigator.share({ files: [file] });
            return;
          } catch (err) {
            // AbortError means the user dismissed the share sheet — leave it
            // at that rather than dropping into a surprise download. Any
            // other failure (e.g. share not actually wired up on this
            // platform despite feature detection) falls through below.
            if (err instanceof Error && err.name === "AbortError") return;
          }
        }

        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      },
      "image/jpeg",
      0.95,
    );
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
          className="absolute inset-0 m-auto block max-h-full max-w-full rounded-lg object-contain shadow-2xl"
        />

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

          <button
            type="button"
            onClick={handleSaveImage}
            className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 transition-colors hover:bg-gold-400"
          >
            {supportsFileShare ? "Save to Photos" : "Download Image"}
          </button>
        </div>
      )}

      {error && <p className="rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-400">{error}</p>}
    </div>
  );
}
