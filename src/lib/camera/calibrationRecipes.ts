import type { BaseFilmSimulation, Recipe } from "@/types/recipe";

/**
 * Synthetic "pure film simulation" recipes for LUT calibration — every field
 * neutral (WB Auto with zero shift, DR100, flat tone/color/sharpness, every
 * effect Off) except baseFilmSimulation. Converting the same source RAF
 * through each of these via CameraLinkContext's convertWithRecipe isolates
 * just the film-simulation's own color transform from the WB shift/tone
 * curve/saturation/grain/Color Chrome math the WebGL shader already applies
 * separately (see fragmentShader.ts's pipeline order — the LUT step is
 * first/isolated, exactly matching this). See the plan doc
 * (~/.claude/plans/indexed-inventing-wren.md) for the full calibration
 * methodology this feeds into.
 *
 * `slug` matches the filename convention already used by
 * scripts/generate-placeholder-luts.mjs and src/engine/webgl/lut.ts's
 * LUT_MANIFEST (e.g. "classic-chrome"), so calibration exports and the
 * derived LUT PNGs line up by name without any extra mapping step.
 */
export interface CalibrationRecipe {
  slug: string;
  recipe: Recipe;
}

function neutralRecipe(baseFilmSimulation: BaseFilmSimulation, slug: string): CalibrationRecipe {
  return {
    slug,
    recipe: {
      id: `calibration-${slug}`,
      name: `Calibration: ${baseFilmSimulation}`,
      baseFilmSimulation,
      dynamicRange: "DR100",
      whiteBalance: { mode: "Auto", shift: { red: 0, blue: 0 } },
      highlightTone: 0,
      shadowTone: 0,
      color: 0,
      sharpness: 0,
      colorChromeEffect: "Off",
      colorChromeFxBlue: "Off",
      grainEffect: "Off",
      clarity: 0,
      compatibleSensors: [],
    },
  };
}

// Phase 1 prototype set (shipped) — a spread across subtle/desaturated
// (Classic Chrome), punchy/saturated (Velvia), and neutral reference
// (Provia). Phase 2 below adds the remaining 11 BaseFilmSimulation values;
// the capture tool and derivation script both loop over whatever's in this
// list, so this is the only place that needs updating.
export const CALIBRATION_RECIPES: CalibrationRecipe[] = [
  neutralRecipe("Classic Chrome", "classic-chrome"),
  neutralRecipe("Velvia", "velvia"),
  neutralRecipe("Provia", "provia"),

  // Phase 2 — the rest of the 14 BaseFilmSimulation values. Slugs match
  // src/engine/webgl/lut.ts's LUT_MANIFEST exactly.
  neutralRecipe("Astia", "astia"),
  neutralRecipe("Pro Neg Hi", "pro-neg-hi"),
  neutralRecipe("Pro Neg Std", "pro-neg-std"),
  neutralRecipe("Classic Negative", "classic-negative"),
  neutralRecipe("Eterna", "eterna"),
  neutralRecipe("Eterna Bleach Bypass", "eterna-bleach-bypass"),
  neutralRecipe("Nostalgic Neg", "nostalgic-neg"),
  neutralRecipe("Reala Ace", "reala-ace"),
  neutralRecipe("Acros", "acros"),
  neutralRecipe("Monochrome", "monochrome"),
  neutralRecipe("Sepia", "sepia"),
];
