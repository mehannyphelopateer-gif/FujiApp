import { useMemo, useRef } from "react";
import type { Recipe } from "@/types/recipe";
import { useWebGLRenderer } from "@/engine/webgl/useWebGLRenderer";
import { computeRecipeAdjustment } from "@/lib/recipes/neutralize";

// One fixed photo, bundled with the site (not user-uploaded, same for every
// visitor) so anyone can see what a recipe actually looks like before
// touching their own photos. No EXIF/camera settings apply to it, so
// computeRecipeAdjustment(null, recipe) just returns the recipe's raw
// values — exactly what we want for a from-scratch comparison render.
const SAMPLE_PHOTO_URL = "/sample-photo/base.jpg";

interface RecipeLivePreviewProps {
  recipe: Recipe;
}

export function RecipeLivePreview({ recipe }: RecipeLivePreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const adjustment = useMemo(() => computeRecipeAdjustment(null, recipe), [recipe]);
  const { isReady, error } = useWebGLRenderer(canvasRef, SAMPLE_PHOTO_URL, adjustment);

  return (
    <div className="relative flex aspect-[3/2] w-full items-center justify-center overflow-hidden rounded-md bg-black">
      <canvas ref={canvasRef} className="max-h-full max-w-full object-contain" />

      {!isReady && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60">
          <svg className="h-6 w-6 animate-spin text-gold-400" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
          </svg>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70 p-3 text-center text-xs text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}
