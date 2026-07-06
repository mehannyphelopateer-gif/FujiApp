import { useCallback, useEffect, useState } from "react";
import type { Recipe } from "@/types/recipe";

const STORAGE_KEY = "fujiapp:custom-recipes";

function loadCustomRecipes(): Recipe[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Recipe[]) : [];
  } catch {
    return [];
  }
}

export function useCustomRecipes() {
  const [customRecipes, setCustomRecipes] = useState<Recipe[]>(loadCustomRecipes);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(customRecipes));
  }, [customRecipes]);

  const saveRecipe = useCallback((recipe: Recipe) => {
    setCustomRecipes((prev) => {
      const index = prev.findIndex((r) => r.id === recipe.id);
      if (index >= 0) {
        const next = [...prev];
        next[index] = recipe;
        return next;
      }
      return [...prev, recipe];
    });
  }, []);

  const deleteRecipe = useCallback((id: string) => {
    setCustomRecipes((prev) => prev.filter((r) => r.id !== id));
  }, []);

  const setPreviewImage = useCallback((id: string, previewImage: string) => {
    setCustomRecipes((prev) => prev.map((r) => (r.id === id ? { ...r, previewImage } : r)));
  }, []);

  return { customRecipes, saveRecipe, deleteRecipe, setPreviewImage };
}
