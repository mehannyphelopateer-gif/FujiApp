import type { Recipe } from "@/types/recipe";

interface RecipeCardProps {
  recipe: Recipe;
  isSelected: boolean;
  onSelect: (id: string) => void;
  isFavorite: boolean;
  onToggleFavorite: (id: string) => void;
}

export function RecipeCard({ recipe, isSelected, onSelect, isFavorite, onToggleFavorite }: RecipeCardProps) {
  return (
    <div
      role="radio"
      aria-checked={isSelected}
      tabIndex={0}
      data-recipe-id={recipe.id}
      onClick={() => onSelect(recipe.id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(recipe.id);
        }
      }}
      className={`cursor-pointer overflow-hidden rounded-md border text-left transition-colors ${
        isSelected ? "border-gold-500 bg-gold-500/10" : "border-ink-800 bg-ink-900 hover:border-ink-600"
      }`}
    >
      <div className="relative aspect-[3/2] w-full overflow-hidden bg-black">
        {recipe.isCustom && !recipe.previewImage ? (
          <div className="flex h-full w-full items-center justify-center bg-ink-800 text-ink-600">
            <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.4" stroke="currentColor" className="h-6 w-6">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 5h16v14H4z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 15l4-4 4 4 3-3 5 5" />
            </svg>
          </div>
        ) : (
          <img
            src={recipe.previewImage ?? `/recipe-previews/${recipe.id}.jpg`}
            alt=""
            loading="lazy"
            className="h-full w-full object-cover"
          />
        )}
        <button
          type="button"
          aria-pressed={isFavorite}
          aria-label={isFavorite ? `Remove ${recipe.name} from favorites` : `Add ${recipe.name} to favorites`}
          onClick={(event) => {
            event.stopPropagation();
            onToggleFavorite(recipe.id);
          }}
          className={`absolute right-1.5 top-1.5 flex h-5 w-5 items-center justify-center rounded-full text-[11px] backdrop-blur-md transition-all ${
            isFavorite ? "bg-gold-500 text-ink-950" : "bg-black/50 text-white hover:bg-black/70"
          }`}
        >
          {isFavorite ? "★" : "☆"}
        </button>
      </div>
      <div className="p-2.5">
        <p className="truncate text-xs font-bold text-ink-50">{recipe.name}</p>
        <p className="truncate font-mono text-[10px] uppercase tracking-wide text-ink-500">
          {recipe.isCustom ? "Custom" : recipe.baseFilmSimulation}
        </p>
      </div>
    </div>
  );
}
