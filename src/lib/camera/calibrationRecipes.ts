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

/** The shared all-neutral base every calibration recipe starts from — see the module doc comment above. */
const NEUTRAL_BASE: Omit<Recipe, "id" | "name" | "baseFilmSimulation"> = {
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
};

/** Builds a calibration recipe from the neutral base plus whatever fields `overrides` changes. */
function calibrationRecipe(overrides: Partial<Recipe> & { baseFilmSimulation: BaseFilmSimulation }, slug: string): CalibrationRecipe {
  return {
    slug,
    recipe: {
      id: `calibration-${slug}`,
      name: `Calibration: ${slug}`,
      ...NEUTRAL_BASE,
      ...overrides,
    },
  };
}

function neutralRecipe(baseFilmSimulation: BaseFilmSimulation, slug: string): CalibrationRecipe {
  return calibrationRecipe({ baseFilmSimulation }, slug);
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

/**
 * Phase 3 — calibrates the shader's remaining hand-tuned approximations
 * (white balance shift AND mode, highlight/shadow tone, saturation,
 * sharpness, Color Chrome Effect, Color Chrome FX Blue, grain) against the
 * real camera, the same way Phases 1-2 calibrated the film-simulation
 * LUTs. Every entry is Provia + all-neutral except the one field being
 * tested, so scripts/derive-parametric-curves.mjs,
 * scripts/derive-color-chrome-luts.mjs, and scripts/derive-grain-stats.mjs
 * can each pair a real converted JPEG against `calib-provia.jpg` from an
 * EXISTING Phase 1/2 shoot folder — see CalibrationCapture.tsx's doc
 * comment for why this list skips the neutral RAW decode Phase 1/2 needed
 * and must be run against a shoot folder that already has
 * `calib-provia.jpg` in it.
 *
 * Slugs match the filenames the derivation scripts expect exactly — see
 * the plan doc's Phase 3 section for the full design.
 */
export const PARAMETRIC_CALIBRATION_RECIPES: CalibrationRecipe[] = [
  // --- White balance shift: red axis (blue held at 0) ---
  calibrationRecipe({ baseFilmSimulation: "Provia", whiteBalance: { mode: "Auto", shift: { red: -9, blue: 0 } } }, "wb-red-m9"),
  calibrationRecipe({ baseFilmSimulation: "Provia", whiteBalance: { mode: "Auto", shift: { red: -4, blue: 0 } } }, "wb-red-m4"),
  calibrationRecipe({ baseFilmSimulation: "Provia", whiteBalance: { mode: "Auto", shift: { red: 4, blue: 0 } } }, "wb-red-p4"),
  calibrationRecipe({ baseFilmSimulation: "Provia", whiteBalance: { mode: "Auto", shift: { red: 9, blue: 0 } } }, "wb-red-p9"),

  // --- White balance shift: blue axis (red held at 0) ---
  calibrationRecipe({ baseFilmSimulation: "Provia", whiteBalance: { mode: "Auto", shift: { red: 0, blue: -9 } } }, "wb-blue-m9"),
  calibrationRecipe({ baseFilmSimulation: "Provia", whiteBalance: { mode: "Auto", shift: { red: 0, blue: -4 } } }, "wb-blue-m4"),
  calibrationRecipe({ baseFilmSimulation: "Provia", whiteBalance: { mode: "Auto", shift: { red: 0, blue: 4 } } }, "wb-blue-p4"),
  calibrationRecipe({ baseFilmSimulation: "Provia", whiteBalance: { mode: "Auto", shift: { red: 0, blue: 9 } } }, "wb-blue-p9"),

  // --- Highlight tone (shadowTone held at 0) ---
  calibrationRecipe({ baseFilmSimulation: "Provia", highlightTone: -2 }, "highlight-m2"),
  calibrationRecipe({ baseFilmSimulation: "Provia", highlightTone: 2 }, "highlight-p2"),
  calibrationRecipe({ baseFilmSimulation: "Provia", highlightTone: 4 }, "highlight-p4"),

  // --- Shadow tone (highlightTone held at 0) ---
  calibrationRecipe({ baseFilmSimulation: "Provia", shadowTone: -2 }, "shadow-m2"),
  calibrationRecipe({ baseFilmSimulation: "Provia", shadowTone: 2 }, "shadow-p2"),
  calibrationRecipe({ baseFilmSimulation: "Provia", shadowTone: 4 }, "shadow-p4"),

  // --- Saturation (Color) ---
  calibrationRecipe({ baseFilmSimulation: "Provia", color: -4 }, "saturation-m4"),
  calibrationRecipe({ baseFilmSimulation: "Provia", color: -2 }, "saturation-m2"),
  calibrationRecipe({ baseFilmSimulation: "Provia", color: 2 }, "saturation-p2"),
  calibrationRecipe({ baseFilmSimulation: "Provia", color: 4 }, "saturation-p4"),

  // --- Color Chrome Effect / FX Blue: isolated, one at a time ---
  calibrationRecipe({ baseFilmSimulation: "Provia", colorChromeEffect: "Weak" }, "cce-weak"),
  calibrationRecipe({ baseFilmSimulation: "Provia", colorChromeEffect: "Strong" }, "cce-strong"),
  calibrationRecipe({ baseFilmSimulation: "Provia", colorChromeFxBlue: "Weak" }, "fxblue-weak"),
  calibrationRecipe({ baseFilmSimulation: "Provia", colorChromeFxBlue: "Strong" }, "fxblue-strong"),
  // Held-out validation only — NOT fit into anything. Checks that composing
  // the two independently-fit LUTs sequentially (the way the shader already
  // does) actually matches a real photo with both effects on at once.
  calibrationRecipe(
    { baseFilmSimulation: "Provia", colorChromeEffect: "Strong", colorChromeFxBlue: "Strong" },
    "cce-strong-fxblue-strong",
  ),

  // --- Grain: all 4 real discrete strength x size combinations ---
  calibrationRecipe({ baseFilmSimulation: "Provia", grainEffect: "Weak", grainSize: "Small" }, "grain-weak-small"),
  calibrationRecipe({ baseFilmSimulation: "Provia", grainEffect: "Strong", grainSize: "Small" }, "grain-strong-small"),
  calibrationRecipe({ baseFilmSimulation: "Provia", grainEffect: "Weak", grainSize: "Large" }, "grain-weak-large"),
  calibrationRecipe({ baseFilmSimulation: "Provia", grainEffect: "Strong", grainSize: "Large" }, "grain-strong-large"),

  // --- Sharpness ---
  calibrationRecipe({ baseFilmSimulation: "Provia", sharpness: -4 }, "sharpness-m4"),
  calibrationRecipe({ baseFilmSimulation: "Provia", sharpness: -2 }, "sharpness-m2"),
  calibrationRecipe({ baseFilmSimulation: "Provia", sharpness: 2 }, "sharpness-p2"),
  calibrationRecipe({ baseFilmSimulation: "Provia", sharpness: 4 }, "sharpness-p4"),

  // --- White balance MODE (shift held at 0, one mode at a time) — "Auto"
  // is the baseline every other recipe here already uses, and "Kelvin"
  // is a continuous temperature dial rather than a fixed preset, so it's
  // out of scope for this pass (see the plan doc's Phase 3 addendum).
  calibrationRecipe({ baseFilmSimulation: "Provia", whiteBalance: { mode: "Daylight", shift: { red: 0, blue: 0 } } }, "wbmode-daylight"),
  calibrationRecipe({ baseFilmSimulation: "Provia", whiteBalance: { mode: "Shade", shift: { red: 0, blue: 0 } } }, "wbmode-shade"),
  calibrationRecipe(
    { baseFilmSimulation: "Provia", whiteBalance: { mode: "Fluorescent1", shift: { red: 0, blue: 0 } } },
    "wbmode-fluorescent1",
  ),
  calibrationRecipe(
    { baseFilmSimulation: "Provia", whiteBalance: { mode: "Fluorescent2", shift: { red: 0, blue: 0 } } },
    "wbmode-fluorescent2",
  ),
  calibrationRecipe(
    { baseFilmSimulation: "Provia", whiteBalance: { mode: "Fluorescent3", shift: { red: 0, blue: 0 } } },
    "wbmode-fluorescent3",
  ),
  calibrationRecipe(
    { baseFilmSimulation: "Provia", whiteBalance: { mode: "Incandescent", shift: { red: 0, blue: 0 } } },
    "wbmode-incandescent",
  ),
  calibrationRecipe(
    { baseFilmSimulation: "Provia", whiteBalance: { mode: "Underwater", shift: { red: 0, blue: 0 } } },
    "wbmode-underwater",
  ),
];
