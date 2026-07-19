import { useMemo, useState } from "react";
import { useAppState } from "@/context/AppStateContext";
import { useFavorites } from "@/hooks/useFavorites";
import { RecipeCard } from "@/components/recipes/RecipeCard";
import type { Recipe } from "@/types/recipe";

type ViewMode = "all" | "favorites";

interface RecipeGridProps {
  /** Overrides the Preview-tab-photo-derived compatible list (e.g. the Camera
   *  tab needs recipes filtered by the actually-connected camera body instead). */
  recipes?: Recipe[];
}

export function RecipeGrid({ recipes: recipesOverride }: RecipeGridProps = {}) {
  const { compatibleRecipes, selectedRecipeId, setSelectedRecipeId } = useAppState();
  const { isFavorite, toggleFavorite } = useFavorites();
  const [view, setView] = useState<ViewMode>("all");

  const sourceRecipes = recipesOverride ?? compatibleRecipes;

  const visibleRecipes = useMemo(() => {
    if (view === "all") return sourceRecipes;
    return sourceRecipes.filter((recipe) => isFavorite(recipe.id));
  }, [sourceRecipes, view, isFavorite]);

  return (
    <div className="space-y-3">
      <div className="flex rounded-md border border-ink-700 bg-ink-900 p-1">
        <button
          type="button"
          onClick={() => setView("all")}
          className={`flex-1 rounded px-2.5 py-1 text-[11px] font-bold uppercase tracking-wide transition-all ${
            view === "all" ? "bg-gold-500 text-ink-950" : "text-ink-400 hover:text-ink-100"
          }`}
        >
          All
        </button>
        <button
          type="button"
          onClick={() => setView("favorites")}
          className={`flex-1 rounded px-2.5 py-1 text-[11px] font-bold uppercase tracking-wide transition-all ${
            view === "favorites" ? "bg-gold-500 text-ink-950" : "text-ink-400 hover:text-ink-100"
          }`}
        >
          ★ Favorites
        </button>
      </div>

      {visibleRecipes.length === 0 ? (
        <p className="py-6 text-center text-xs text-ink-500">
          {view === "favorites" ? "No favorites yet — tap ☆ on a recipe to save it here." : "No recipes match."}
        </p>
      ) : (
        <div role="radiogroup" aria-label="Film simulation recipes" className="grid grid-cols-2 gap-2">
          {visibleRecipes.map((recipe) => (
            <RecipeCard
              key={recipe.id}
              recipe={recipe}
              isSelected={recipe.id === selectedRecipeId}
              onSelect={setSelectedRecipeId}
              isFavorite={isFavorite(recipe.id)}
              onToggleFavorite={toggleFavorite}
            />
          ))}
        </div>
      )}
    </div>
  );
}
