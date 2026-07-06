import { useMemo, useRef } from "react";
import type { Recipe } from "@/types/recipe";
import type { DetectedSettings } from "@/types/exif";
import { computeRecipeAdjustment } from "@/lib/recipes/neutralize";
import { useWebGLRenderer } from "@/engine/webgl/useWebGLRenderer";

// Comparison slots render several canvases at once — each one is shown small,
// so there's no need for the main previewer's full 4096px cap. Capping much
// lower here keeps multiple simultaneous WebGL contexts cheap on memory and
// fast to redraw.
const COMPARE_MAX_DIMENSION = 720;

interface CompareSlotProps {
  recipe: Recipe;
  previewUrl: string;
  detectedSettings: DetectedSettings | null;
  onRemove: () => void;
}

export function CompareSlot({ recipe, previewUrl, detectedSettings, onRemove }: CompareSlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const adjustment = useMemo(() => computeRecipeAdjustment(detectedSettings, recipe), [detectedSettings, recipe]);
  const { error } = useWebGLRenderer(canvasRef, previewUrl, adjustment, COMPARE_MAX_DIMENSION);

  return (
    <div className="space-y-2">
      <div className="relative aspect-square w-full overflow-hidden rounded-md bg-black">
        <canvas ref={canvasRef} className="h-full w-full object-contain" />
        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${recipe.name} from comparison`}
          className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-black/60 text-xs text-white hover:bg-black/80"
        >
          ✕
        </button>
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80 p-2 text-center text-[10px] text-red-400">
            {error}
          </div>
        )}
      </div>
      <div>
        <p className="truncate text-xs font-bold text-ink-50">{recipe.name}</p>
        <p className="truncate font-mono text-[10px] uppercase tracking-wide text-ink-500">
          {recipe.baseFilmSimulation}
        </p>
      </div>
    </div>
  );
}
