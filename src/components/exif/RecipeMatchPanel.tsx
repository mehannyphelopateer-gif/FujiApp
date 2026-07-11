import { useMemo } from "react";
import { useAppState } from "@/context/AppStateContext";
import { findMatchingRecipes } from "@/lib/recipes/matchRecipe";

/**
 * Deterministic counterpart to SceneAnalysisPanel's AI-based suggestion:
 * "which recipe was probably actually used to shoot this" from the photo's
 * detected camera settings (see lib/recipes/matchRecipe.ts), rather than
 * "which recipe would suit this scene." No network call — pure local math
 * against the recipe catalog, so it's instant and always available once a
 * photo with intact Fuji MakerNotes is loaded.
 */
export function RecipeMatchPanel() {
  const { selectedFile, detectedSettings, compatibleRecipes, setSelectedRecipeId } = useAppState();

  const matches = useMemo(() => {
    if (!detectedSettings) return null;
    return findMatchingRecipes(detectedSettings, compatibleRecipes);
  }, [detectedSettings, compatibleRecipes]);

  if (!selectedFile) return null;

  if (!detectedSettings) {
    return (
      <p className="rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-xs text-ink-500">
        No Fuji metadata detected in this image — can't identify which recipe was used.
      </p>
    );
  }

  if (!matches || matches.length === 0) {
    return <p className="text-xs text-ink-500">No recipes to compare against.</p>;
  }

  return (
    <div className="space-y-2">
      {matches.map(({ recipe, score }) => (
        <div key={recipe.id} className="rounded border border-ink-800 bg-ink-900 p-2.5">
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <p className="truncate text-sm font-bold text-ink-50">{recipe.name}</p>
              <p className="font-mono text-[10px] uppercase tracking-wide text-ink-500">{recipe.baseFilmSimulation}</p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <span className="rounded bg-ink-800 px-1.5 py-0.5 font-mono text-[10px] font-bold text-gold-300">
                {score}% match
              </span>
              <button
                type="button"
                onClick={() => setSelectedRecipeId(recipe.id)}
                className="rounded bg-gold-500/15 px-2 py-1 text-[11px] font-bold uppercase text-gold-400 transition-colors hover:bg-gold-500/25"
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
