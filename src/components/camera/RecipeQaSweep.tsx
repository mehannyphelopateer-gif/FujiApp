import { useRef, useState } from "react";
import type { Recipe } from "@/types/recipe";
import { useCameraLink } from "@/context/CameraLinkContext";

interface SweepResult {
  recipeId: string;
  name: string;
  status: "ok" | "error";
  error?: string;
  thumbnailUrl?: string;
}

/**
 * Downscales the full-res converted JPEG to a small data URL for the results
 * grid — keeping ~120 full-res camera JPEGs (~4MB each) in memory at once
 * isn't necessary just to eyeball them for obvious problems, and would risk
 * a WebView memory crash on a long sweep.
 */
function createThumbnail(url: string, maxDim = 320): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas not supported."));
        return;
      }
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", 0.7));
    };
    img.onerror = () => reject(new Error("Failed to decode the converted image."));
    img.src = url;
  });
}

interface RecipeQaSweepProps {
  rafFile: File;
  recipes: Recipe[];
}

/**
 * Runs every recipe through the exact same convertWithRecipe pipeline real
 * usage does (upload -> patch -> trigger -> poll -> download), sequentially,
 * against one already-loaded RAF — a mechanical pass/fail sweep (did the
 * camera accept the profile and produce a JPEG, or did it error/time out),
 * not a substitute for actually looking at the results: a recipe can convert
 * "successfully" and still look wrong (e.g. the WB/FX-Blue color-cast case
 * found earlier this session), so the thumbnail grid is there to look at.
 */
export function RecipeQaSweep({ rafFile, recipes }: RecipeQaSweepProps) {
  const { convertWithRecipe } = useCameraLink();
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState<{ index: number; name: string } | null>(null);
  const [results, setResults] = useState<SweepResult[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const cancelRef = useRef(false);

  async function handleRun() {
    setIsRunning(true);
    setResults([]);
    setExpanded(null);
    cancelRef.current = false;

    for (let i = 0; i < recipes.length; i++) {
      if (cancelRef.current) break;
      const recipe = recipes[i];
      setProgress({ index: i + 1, name: recipe.name });

      const outcome = await convertWithRecipe(recipe, rafFile);
      let entry: SweepResult;
      if (outcome.ok && outcome.imageUrl) {
        try {
          const thumbnailUrl = await createThumbnail(outcome.imageUrl);
          entry = { recipeId: recipe.id, name: recipe.name, status: "ok", thumbnailUrl };
        } catch (err) {
          entry = {
            recipeId: recipe.id,
            name: recipe.name,
            status: "error",
            error: err instanceof Error ? err.message : "Couldn't build a thumbnail.",
          };
        }
      } else {
        entry = { recipeId: recipe.id, name: recipe.name, status: "error", error: outcome.error };
      }
      setResults((prev) => [...prev, entry]);
    }

    setIsRunning(false);
    setProgress(null);
  }

  const okCount = results.filter((r) => r.status === "ok").length;
  const errorCount = results.filter((r) => r.status === "error").length;

  return (
    <div className="space-y-3">
      <p className="text-[11px] text-ink-500">
        Runs every compatible recipe against the loaded RAF, one at a time, and reports which ones the camera
        accepted. This takes a while (roughly 10s per recipe) — it re-uploads the RAF fresh for each one, the same
        way a real recipe switch does. Catches mechanical failures (camera rejects a value, times out); still worth
        eyeballing the thumbnails yourself for anything that just looks wrong.
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={handleRun}
          disabled={isRunning}
          className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isRunning ? "Running…" : `Run QA Sweep (${recipes.length} recipes)`}
        </button>
        {isRunning && (
          <button
            type="button"
            onClick={() => {
              cancelRef.current = true;
            }}
            className="rounded-md border border-red-800 px-4 py-2 text-xs font-bold uppercase tracking-wide text-red-400"
          >
            Cancel
          </button>
        )}
      </div>

      {progress && (
        <p className="text-[11px] font-bold uppercase tracking-wide text-ink-400">
          Testing {progress.index}/{recipes.length} — {progress.name}…
        </p>
      )}

      {results.length > 0 && (
        <>
          <p className="text-[11px] font-bold uppercase tracking-wide text-ink-400">
            <span className="text-green-400">{okCount} ok</span> · <span className="text-red-400">{errorCount} errors</span>
          </p>
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
            {results.map((result) => (
              <button
                key={result.recipeId}
                type="button"
                onClick={() => setExpanded(expanded === result.recipeId ? null : result.recipeId)}
                className={`overflow-hidden rounded-md border text-left ${
                  result.status === "error" ? "border-red-700" : "border-ink-800"
                }`}
              >
                <div className="flex aspect-square items-center justify-center bg-black/30">
                  {result.thumbnailUrl ? (
                    <img src={result.thumbnailUrl} alt={result.name} className="h-full w-full object-cover" />
                  ) : (
                    <span className="p-1 text-center text-[10px] text-red-400">Error</span>
                  )}
                </div>
                <p className="truncate bg-ink-900 px-1.5 py-1 text-[10px] text-ink-300">{result.name}</p>
              </button>
            ))}
          </div>
          {expanded && (
            <div className="rounded-md border border-ink-800 bg-ink-900 p-3">
              {(() => {
                const result = results.find((r) => r.recipeId === expanded);
                if (!result) return null;
                return (
                  <div className="space-y-2">
                    <p className="text-xs font-bold text-ink-50">{result.name}</p>
                    {result.thumbnailUrl && (
                      <img src={result.thumbnailUrl} alt={result.name} className="max-h-64 w-full rounded-md object-contain" />
                    )}
                    {result.error && <p className="text-[11px] text-red-400">{result.error}</p>}
                  </div>
                );
              })()}
            </div>
          )}
        </>
      )}
    </div>
  );
}
