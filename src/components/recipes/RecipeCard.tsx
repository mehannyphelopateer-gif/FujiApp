import type { Recipe } from "@/types/recipe";

interface RecipeCardProps {
  recipe: Recipe;
  isSelected: boolean;
  onSelect: (id: string) => void;
}

export function RecipeCard({ recipe, isSelected, onSelect }: RecipeCardProps) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={isSelected}
      data-recipe-id={recipe.id}
      onClick={() => onSelect(recipe.id)}
      className={`overflow-hidden rounded-md border text-left transition-colors ${
        isSelected ? "border-gold-500 bg-gold-500/10" : "border-ink-800 bg-ink-900 hover:border-ink-600"
      }`}
    >
      <div className="aspect-[3/2] w-full overflow-hidden bg-black">
        <img
          src={`/recipe-previews/${recipe.id}.jpg`}
          alt=""
          loading="lazy"
          className="h-full w-full object-cover"
        />
      </div>
      <div className="p-2.5">
        <p className="truncate text-xs font-bold text-ink-50">{recipe.name}</p>
        <p className="truncate font-mono text-[10px] uppercase tracking-wide text-ink-500">
          {recipe.baseFilmSimulation}
        </p>
      </div>
    </button>
  );
}
