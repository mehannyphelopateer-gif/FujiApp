import { useAppState } from "@/context/AppStateContext";
import { RecipeCard } from "@/components/recipes/RecipeCard";

export function RecipeGrid() {
  const { compatibleRecipes, selectedRecipeId, setSelectedRecipeId } = useAppState();

  return (
    <div role="radiogroup" aria-label="Film simulation recipes" className="grid grid-cols-2 gap-2">
      {compatibleRecipes.map((recipe) => (
        <RecipeCard
          key={recipe.id}
          recipe={recipe}
          isSelected={recipe.id === selectedRecipeId}
          onSelect={setSelectedRecipeId}
        />
      ))}
    </div>
  );
}
