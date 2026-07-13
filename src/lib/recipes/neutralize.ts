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
  colorChromeEffect: "Off",
  colorChromeFxBlue: "Off",
  grainEffect: "Off",
};

// Off/Weak/Strong as ordinal fractions, so a partially-neutralized value
// (target weaker than what's already baked in) can land between them.
const EFFECT_STRENGTH_FRACTION: Record<EffectStrength, number> = { Off: 0, Weak: 0.5, Strong: 1 };

/**
 * Grain/Color Chrome/FX Blue are baked into the image's actual pixels (real
 * noise, real luminance compression) exactly like the tone curve is — but
 * unlike WB/tone/color, there's no plausible "negative" amount of grain to
 * request, so a straight `target - detected` delta could go negative and
 * mean nothing to the shader. Clamped at 0: if the photo already has more
 * of an effect baked in than the target recipe wants, we add none on top
 * (can't be undone either way); if it wants more, we add the difference.
 */
function neutralizedStrength(detected: EffectStrength, target: EffectStrength): number {
  return Math.max(0, EFFECT_STRENGTH_FRACTION[target] - EFFECT_STRENGTH_FRACTION[detected]);
}

export interface RecipeAdjustment {
  whiteBalanceShift: { red: number; blue: number }; // neutralized delta
  highlightTone: number; // neutralized delta
  shadowTone: number; // neutralized delta
  color: number; // neutralized delta (saturation)
  sharpness: number; // forward-only target value, NOT a delta
  colorChromeStrength: number; // neutralized 0..1 fraction (0=Off, 0.5=Weak, 1=Strong)
  colorChromeFxBlueStrength: number; // neutralized 0..1 fraction
  grainStrength: number; // neutralized 0..1 fraction
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
 * applySharpness, which targets this value directly).
 *
 * Color Chrome Effect/FX Blue and Grain get the same neutralization
 * treatment as tone/color (see neutralizedStrength above) rather than being
 * forwarded straight from the target recipe — a RAF's embedded preview (or
 * any Fuji JPEG) already has whatever effect strength was dialed in at
 * capture baked into the actual pixels, so applying a second recipe's
 * grain/color-chrome on top of that unconditionally would stack rather than
 * replace it (e.g. a heavy-grain recipe's photo would still look grainy
 * after switching to a no-grain recipe's preview). Grain size has no
 * meaningful "zero" to delta against, so it's still forwarded from the
 * target directly — it only matters once grainStrength is actually > 0.
 *
 * noiseReduction/isoRange/exposureCompensation/clarity are deliberately
 * excluded here — they're capture-time camera settings (or, for clarity,
 * simply not wired into the shader at all), not something a post-process
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
    colorChromeStrength: neutralizedStrength(baseline.colorChromeEffect, target.colorChromeEffect),
    colorChromeFxBlueStrength: neutralizedStrength(baseline.colorChromeFxBlue, target.colorChromeFxBlue),
    grainStrength: neutralizedStrength(baseline.grainEffect, target.grainEffect),
    grainSize: target.grainSize,
    baseFilmSimulation: target.baseFilmSimulation,
  };
}
