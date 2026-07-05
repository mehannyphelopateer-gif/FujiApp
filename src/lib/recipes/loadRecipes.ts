import type { Recipe } from "@/types/recipe";
import recipesData from "@/data/recipes.json";

const REQUIRED_FIELDS: (keyof Recipe)[] = [
  "id",
  "name",
  "baseFilmSimulation",
  "dynamicRange",
  "whiteBalance",
  "highlightTone",
  "shadowTone",
  "color",
  "sharpness",
  "colorChromeEffect",
  "colorChromeFxBlue",
  "grainEffect",
  "compatibleSensors",
];

/**
 * `as Recipe[]` alone doesn't catch missing required fields — TypeScript's
 * structural checking on a type assertion doesn't reject a JSON literal
 * that's simply missing a property the interface requires, so a recipe
 * silently missing e.g. `color` would compile fine and only break at
 * render time (NaN uniforms, "undefined" in the sidebar). This is a cheap
 * runtime check to fail loudly at load time instead, since recipes.json is
 * large and hand/programmatically authored, not hand-verified line by line.
 */
function validateRecipes(data: unknown): Recipe[] {
  if (!Array.isArray(data)) throw new Error("recipes.json must be an array");

  data.forEach((entry, index) => {
    const missing = REQUIRED_FIELDS.filter((field) => entry[field] === undefined);
    if (missing.length > 0) {
      const label = entry.id ?? entry.name ?? `index ${index}`;
      throw new Error(`recipes.json entry "${label}" is missing required field(s): ${missing.join(", ")}`);
    }
  });

  return data as Recipe[];
}

export const recipes: Recipe[] = validateRecipes(recipesData);

export function getRecipeById(id: string): Recipe | undefined {
  return recipes.find((recipe) => recipe.id === id);
}
