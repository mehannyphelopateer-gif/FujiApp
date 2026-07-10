import type { Recipe } from "@/types/recipe";

export interface DialInStep {
  id: string;
  menuPath: string;
  label: string;
  value: string;
}

function formatSigned(value: number): string {
  return value > 0 ? `+${value}` : `${value}`;
}

/**
 * Walks a Recipe's fields in the same order they appear in a Fuji camera's
 * IMAGE QUALITY SETTING menu, so the guide can be followed top-to-bottom
 * without jumping around the camera's own UI. Skips fields that don't apply
 * (e.g. WB shift when it's zero, grain size when grain is Off).
 */
export function buildDialInSteps(recipe: Recipe): DialInStep[] {
  const { whiteBalance } = recipe;
  const steps: DialInStep[] = [
    { id: "film-sim", menuPath: "IMAGE QUALITY SETTING > FILM SIMULATION", label: "Film Simulation", value: recipe.baseFilmSimulation },
    { id: "dynamic-range", menuPath: "IMAGE QUALITY SETTING > D RANGE", label: "Dynamic Range", value: recipe.dynamicRange },
    {
      id: "wb-mode",
      menuPath: "IMAGE QUALITY SETTING > WHITE BALANCE",
      label: "White Balance Mode",
      value: whiteBalance.mode === "Kelvin" && whiteBalance.kelvin ? `${whiteBalance.kelvin}K (Kelvin)` : whiteBalance.mode,
    },
  ];

  if (whiteBalance.shift.red !== 0 || whiteBalance.shift.blue !== 0) {
    steps.push({
      id: "wb-shift",
      menuPath: "IMAGE QUALITY SETTING > WHITE BALANCE > (fine-tune after selecting mode)",
      label: "White Balance Shift",
      value: `Red ${formatSigned(whiteBalance.shift.red)}, Blue ${formatSigned(whiteBalance.shift.blue)}`,
    });
  }

  steps.push(
    { id: "highlight", menuPath: "IMAGE QUALITY SETTING > TONE CURVE > HIGHLIGHT", label: "Highlight Tone", value: formatSigned(recipe.highlightTone) },
    { id: "shadow", menuPath: "IMAGE QUALITY SETTING > TONE CURVE > SHADOW", label: "Shadow Tone", value: formatSigned(recipe.shadowTone) },
    { id: "color", menuPath: "IMAGE QUALITY SETTING > COLOR", label: "Color", value: formatSigned(recipe.color) },
    { id: "sharpness", menuPath: "IMAGE QUALITY SETTING > SHARPNESS", label: "Sharpness", value: formatSigned(recipe.sharpness) },
    { id: "color-chrome", menuPath: "IMAGE QUALITY SETTING > COLOR CHROME EFFECT", label: "Color Chrome Effect", value: recipe.colorChromeEffect },
    { id: "color-chrome-fx-blue", menuPath: "IMAGE QUALITY SETTING > COLOR CHROME FX BLUE", label: "Color Chrome FX Blue", value: recipe.colorChromeFxBlue },
    { id: "grain-effect", menuPath: "IMAGE QUALITY SETTING > GRAIN EFFECT > STRENGTH", label: "Grain Effect", value: recipe.grainEffect },
  );

  if (recipe.grainEffect !== "Off" && recipe.grainSize) {
    steps.push({ id: "grain-size", menuPath: "IMAGE QUALITY SETTING > GRAIN EFFECT > SIZE", label: "Grain Size", value: recipe.grainSize });
  }

  if (recipe.noiseReduction !== undefined) {
    steps.push({ id: "noise-reduction", menuPath: "IMAGE QUALITY SETTING > NOISE REDUCTION", label: "Noise Reduction", value: formatSigned(recipe.noiseReduction) });
  }
  if (recipe.clarity !== undefined) {
    steps.push({ id: "clarity", menuPath: "IMAGE QUALITY SETTING > CLARITY", label: "Clarity", value: formatSigned(recipe.clarity) });
  }
  if (recipe.isoRange !== undefined) {
    steps.push({ id: "iso", menuPath: "ISO (shooting setting, not in this menu)", label: "ISO", value: recipe.isoRange });
  }
  if (recipe.exposureCompensation !== undefined) {
    steps.push({
      id: "exposure-compensation",
      menuPath: "Exposure Compensation dial (shooting setting, not in this menu)",
      label: "Exposure Compensation",
      value:
        typeof recipe.exposureCompensation === "number"
          ? `${recipe.exposureCompensation > 0 ? "+" : ""}${recipe.exposureCompensation} EV`
          : recipe.exposureCompensation,
    });
  }

  return steps;
}
