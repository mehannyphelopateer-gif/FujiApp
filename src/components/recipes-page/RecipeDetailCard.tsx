import { useState } from "react";
import type { Recipe } from "@/types/recipe";
import { RecipeParameterList } from "@/components/recipes/RecipeParameterList";

interface RecipeDetailCardProps {
  recipe: Recipe;
  isFavorite: boolean;
  onToggleFavorite: (id: string) => void;
}

export function RecipeDetailCard({ recipe, isFavorite, onToggleFavorite }: RecipeDetailCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="group overflow-hidden rounded-lg border border-ink-800 bg-ink-900 transition-colors hover:border-gold-700/60">
      <div className="relative aspect-[3/2] w-full overflow-hidden bg-black">
        <img
          src={`/recipe-previews/${recipe.id}.jpg`}
          alt={`Example photo with the ${recipe.name} recipe applied`}
          loading="lazy"
          className="h-full w-full object-cover transition-transform duration-500 ease-out group-hover:scale-105"
        />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/85 via-black/10 to-transparent" />

        <button
          type="button"
          aria-pressed={isFavorite}
          aria-label={isFavorite ? `Remove ${recipe.name} from favorites` : `Add ${recipe.name} to favorites`}
          onClick={() => onToggleFavorite(recipe.id)}
          className={`absolute right-2.5 top-2.5 flex h-7 w-7 items-center justify-center rounded-full text-sm backdrop-blur-md transition-all ${
            isFavorite ? "bg-gold-500 text-ink-950" : "bg-black/50 text-white hover:bg-black/70"
          }`}
        >
          {isFavorite ? "★" : "☆"}
        </button>

        <div className="absolute inset-x-0 bottom-0 p-3.5">
          <p className="text-base font-black uppercase leading-tight tracking-tight text-white">{recipe.name}</p>
          <p className="font-mono text-[10px] font-semibold uppercase tracking-wider text-gold-400">
            {recipe.baseFilmSimulation}
          </p>
        </div>
      </div>

      <div className="p-3.5">
        {recipe.description && (
          <p className="line-clamp-2 text-xs leading-relaxed text-ink-400">{recipe.description}</p>
        )}

        <button
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
          className="mt-3 flex items-center gap-1 text-xs font-bold uppercase tracking-wide text-gold-400 transition-colors hover:text-gold-300"
        >
          {expanded ? "Hide settings" : "Show settings"}
          <svg
            viewBox="0 0 24 24"
            fill="none"
            strokeWidth="2.5"
            stroke="currentColor"
            className={`h-3 w-3 transition-transform ${expanded ? "rotate-180" : ""}`}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 9l6 6 6-6" />
          </svg>
        </button>

        {expanded && (
          <div className="mt-4 border-t border-ink-800 pt-4">
            <RecipeParameterList recipe={recipe} />
          </div>
        )}
      </div>
    </div>
  );
}
