import { useMemo, useState } from "react";
import { recipes } from "@/lib/recipes/loadRecipes";
import { useFavorites } from "@/hooks/useFavorites";
import { RecipeDetailCard } from "@/components/recipes-page/RecipeDetailCard";

type ViewMode = "all" | "favorites";

export function RecipesPage() {
  const [search, setSearch] = useState("");
  const [view, setView] = useState<ViewMode>("all");
  const { isFavorite, toggleFavorite } = useFavorites();

  const filteredRecipes = useMemo(() => {
    const query = search.trim().toLowerCase();
    return recipes.filter((recipe) => {
      if (view === "favorites" && !isFavorite(recipe.id)) return false;
      if (!query) return true;
      return (
        recipe.name.toLowerCase().includes(query) ||
        recipe.baseFilmSimulation.toLowerCase().includes(query) ||
        (recipe.description?.toLowerCase().includes(query) ?? false)
      );
    });
  }, [search, view, isFavorite]);

  return (
    <div className="h-full w-full overflow-y-auto bg-ink-950 text-ink-50">
      <header className="border-b border-ink-800 px-4 py-6 [padding-top:calc(1.5rem+env(safe-area-inset-top))]">
        <div className="mb-1.5 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.15em] text-gold-500">
          <span className="h-px w-3 bg-gold-600" />
          Recipe Library
        </div>
        <h1 className="text-4xl font-black uppercase leading-[0.95] tracking-tight text-ink-50">
          All Recipes<span className="text-gold-400">.</span>
        </h1>
        <p className="mt-2 text-xs text-ink-400">
          {recipes.length} film simulation recipes, rendered with the app's own engine.
        </p>
      </header>

      <div className="sticky top-0 z-10 flex flex-col gap-3 border-b border-ink-800 bg-ink-950/95 px-4 py-3 backdrop-blur-md sm:flex-row sm:items-center">
        <div className="flex rounded-md border border-ink-700 bg-ink-900 p-1">
          <button
            type="button"
            onClick={() => setView("all")}
            className={`rounded px-3.5 py-1.5 text-xs font-bold uppercase tracking-wide transition-all ${
              view === "all" ? "bg-gold-500 text-ink-950" : "text-ink-400 hover:text-ink-100"
            }`}
          >
            All Recipes
          </button>
          <button
            type="button"
            onClick={() => setView("favorites")}
            className={`rounded px-3.5 py-1.5 text-xs font-bold uppercase tracking-wide transition-all ${
              view === "favorites" ? "bg-gold-500 text-ink-950" : "text-ink-400 hover:text-ink-100"
            }`}
          >
            ★ Favorites
          </button>
        </div>

        <div className="relative flex-1 sm:max-w-xs">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            strokeWidth="2"
            stroke="currentColor"
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-500"
          >
            <circle cx="11" cy="11" r="7" />
            <path strokeLinecap="round" d="M21 21l-4.3-4.3" />
          </svg>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search recipes…"
            className="w-full rounded-md border border-ink-700 bg-ink-900 py-2 pl-9 pr-3 text-sm text-ink-50 placeholder:text-ink-500 transition-colors focus:border-gold-500 focus:outline-none"
          />
        </div>

        <span className="font-mono text-xs font-medium text-ink-500 sm:ml-auto">
          {filteredRecipes.length} / {recipes.length}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 p-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filteredRecipes.map((recipe) => (
          <RecipeDetailCard
            key={recipe.id}
            recipe={recipe}
            isFavorite={isFavorite(recipe.id)}
            onToggleFavorite={toggleFavorite}
          />
        ))}
        {filteredRecipes.length === 0 && (
          <p className="col-span-full py-16 text-center text-sm text-ink-500">
            {view === "favorites" ? "No favorites yet — tap the star on a recipe to save it here." : "No recipes match."}
          </p>
        )}
      </div>
    </div>
  );
}
