import type { DetectedSettings } from "@/types/exif";
import type { Recipe } from "@/types/recipe";

export interface RecipeMatch {
  recipe: Recipe;
  /** 0-100, higher = closer match. A relative ranking signal, not a calibrated probability. */
  score: number;
}

// Normalization ranges, matching each field's actual scale (see types/recipe.ts).
const WB_SHIFT_RANGE = 18; // -9..9
const TONE_RANGE = 6; // -2..4
const COLOR_RANGE = 8; // -4..4
const SHARPNESS_RANGE = 8; // -4..4
const WB_MODE_MISMATCH_PENALTY = 0.3;

// Sum of each term's worst-case normalized contribution — used to scale
// distance into a 0-100 score. Two WB-shift axes + two tone fields + color +
// sharpness, each capped at 1.0 when fully normalized, plus the WB-mode
// mismatch penalty.
const MAX_DISTANCE = 2 + 2 + 1 + 1 + WB_MODE_MISMATCH_PENALTY;

function numericDistance(detected: DetectedSettings, recipe: Recipe): number {
  const wbShiftDiff =
    Math.abs(detected.whiteBalance.shift.red - recipe.whiteBalance.shift.red) / WB_SHIFT_RANGE +
    Math.abs(detected.whiteBalance.shift.blue - recipe.whiteBalance.shift.blue) / WB_SHIFT_RANGE;
  const toneDiff =
    Math.abs(detected.highlightTone - recipe.highlightTone) / TONE_RANGE +
    Math.abs(detected.shadowTone - recipe.shadowTone) / TONE_RANGE;
  const colorDiff = Math.abs(detected.color - recipe.color) / COLOR_RANGE;
  const sharpnessDiff = Math.abs(detected.sharpness - recipe.sharpness) / SHARPNESS_RANGE;
  const wbModeDiff =
    detected.whiteBalance.mode !== "Unknown" && detected.whiteBalance.mode !== recipe.whiteBalance.mode
      ? WB_MODE_MISMATCH_PENALTY
      : 0;

  return wbShiftDiff + toneDiff + colorDiff + sharpnessDiff + wbModeDiff;
}

/**
 * Finds the recipes whose baked-in settings most closely match a photo's
 * detected camera settings — i.e. "which recipe was probably used to shoot
 * this." Film simulation is a hard filter first: a recipe with a different
 * base film simulation could not have produced this photo (it's a discrete,
 * reliably-detected camera setting), whereas WB/tone/color/sharpness are
 * gradations where a recipe author's dial-in and the actual shot can drift
 * slightly. Falls back to the unfiltered pool (with visibly lower scores)
 * if nothing shares the detected film simulation, or if detection itself
 * came back "Unknown" for film simulation entirely.
 */
export function findMatchingRecipes(detected: DetectedSettings, recipes: Recipe[], limit = 3): RecipeMatch[] {
  const filmSimKnown = detected.baseFilmSimulation !== "Unknown";
  const filtered = filmSimKnown ? recipes.filter((recipe) => recipe.baseFilmSimulation === detected.baseFilmSimulation) : recipes;
  const pool = filtered.length > 0 ? filtered : recipes;

  const scored: RecipeMatch[] = pool.map((recipe) => {
    const distance = numericDistance(detected, recipe);
    const score = Math.max(0, Math.round(100 * (1 - distance / MAX_DISTANCE)));
    return { recipe, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}
