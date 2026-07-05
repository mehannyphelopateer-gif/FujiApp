import type { DetectedSettings } from "@/types/exif";
import type { BaseFilmSimulation, EffectStrength, GrainSize, Recipe } from "@/types/recipe";

const NEUTRAL_BASELINE: DetectedSettings = {
  cameraModel: null,
  baseFilmSimulation: "Unknown",
  whiteBalance: { mode: "Unknown", shift: { red: 0, blue: 0 } },
  highlightTone: 0,
  shadowTone: 0,
  color: 0,
  sharpness: 0,
};

export interface RecipeAdjustment {
  whiteBalanceShift: { red: number; blue: number }; // neutralized delta
  highlightTone: number; // neutralized delta
  shadowTone: number; // neutralized delta
  color: number; // neutralized delta (saturation)
  sharpness: number; // forward-only target value, NOT a delta
  colorChromeEffect: EffectStrength;
  colorChromeFxBlue: EffectStrength;
  grainEffect: EffectStrength;
  grainSize?: GrainSize;
  baseFilmSimulation: BaseFilmSimulation;
}

/**
 * Computes what the shader should actually apply: the difference between
 * the target recipe and whatever the camera already baked into the image,
 * so the two don't stack. Falls back to a neutral (all-zero) baseline when
 * no Fuji MakerNotes were found, which makes the delta collapse to the
 * recipe's raw values — i.e. the recipe is applied directly, matching
 * simple "no metadata available" behavior.
 *
 * Sharpness is exempt from the delta math on purpose: reversing baked-in
 * sharpening is a deconvolution problem that produces artifacts in a
 * real-time shader, so it's forward-only (see fragmentShader.ts's
 * applySharpness, which targets this value directly). Color Chrome
 * Effect/FX Blue and Grain are look characteristics, not baked tone
 * curves — they're already enum/forward-only, same as before.
 *
 * noiseReduction/isoRange/exposureCompensation are deliberately excluded
 * here — they're capture-time camera settings, not something a post-process
 * shader can apply to an already-rendered JPEG. See Recipe's doc comment.
 */
export function computeRecipeAdjustment(detected: DetectedSettings | null, target: Recipe): RecipeAdjustment {
  const baseline = detected ?? NEUTRAL_BASELINE;

  return {
    whiteBalanceShift: {
      red: target.whiteBalance.shift.red - baseline.whiteBalance.shift.red,
      blue: target.whiteBalance.shift.blue - baseline.whiteBalance.shift.blue,
    },
    highlightTone: target.highlightTone - baseline.highlightTone,
    shadowTone: target.shadowTone - baseline.shadowTone,
    color: target.color - baseline.color,
    sharpness: target.sharpness,
    colorChromeEffect: target.colorChromeEffect,
    colorChromeFxBlue: target.colorChromeFxBlue,
    grainEffect: target.grainEffect,
    grainSize: target.grainSize,
    baseFilmSimulation: target.baseFilmSimulation,
  };
}
