import { useState } from "react";
import { useAppState } from "@/context/AppStateContext";
import { recommendRecipeForPhoto, type RecommendationResult } from "@/lib/ai/recommendRecipe";

export function SceneAnalysisPanel() {
  const { selectedFile, compatibleRecipes, setSelectedRecipeId } = useAppState();
  const [result, setResult] = useState<RecommendationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!selectedFile) {
    return (
      <div className="flex flex-col items-center gap-2 py-12 text-center">
        <p className="text-sm text-ink-400">Upload a photo on the Preview tab first.</p>
        <p className="text-xs text-ink-600">Then come back here and I'll suggest a recipe for it.</p>
      </div>
    );
  }

  async function handleAnalyze() {
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const recommendation = await recommendRecipeForPhoto(selectedFile!);
      setResult(recommendation);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  const recipeById = new Map(compatibleRecipes.map((r) => [r.id, r]));

  return (
    <div className="space-y-3">
      <button
        type="button"
        onClick={handleAnalyze}
        disabled={isLoading}
        className="flex w-full items-center justify-center gap-2 rounded-md border border-gold-600 bg-gold-500/10 px-3 py-2.5 text-sm font-bold uppercase tracking-wide text-gold-400 transition-all hover:bg-gold-500/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isLoading ? (
          <>
            <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
            </svg>
            Analyzing scene…
          </>
        ) : (
          <>Suggest a recipe for this scene</>
        )}
      </button>

      {error && <p className="rounded-md bg-red-500/10 px-3 py-2 text-xs text-red-400">{error}</p>}

      {result && (
        <div className="space-y-2.5 rounded-md border border-ink-800 bg-ink-900 p-3">
          <p className="text-xs leading-relaxed text-ink-400">{result.sceneDescription}</p>

          <div className="space-y-2">
            {result.recommendations.map((rec) => {
              const recipe = recipeById.get(rec.recipeId);
              if (!recipe) return null;
              return (
                <div key={rec.recipeId} className="rounded border border-ink-800 bg-ink-800/60 p-2.5">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-sm font-bold text-ink-50">{recipe.name}</p>
                    <button
                      type="button"
                      onClick={() => setSelectedRecipeId(recipe.id)}
                      className="shrink-0 rounded bg-gold-500/15 px-2 py-1 text-[11px] font-bold uppercase text-gold-400 transition-colors hover:bg-gold-500/25"
                    >
                      Apply
                    </button>
                  </div>
                  <p className="mt-1 text-xs text-ink-500">{rec.reasoning}</p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
