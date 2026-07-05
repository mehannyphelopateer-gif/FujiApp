import type { Recipe } from "@/types/recipe";
import { ParameterReadout } from "@/components/recipes/ParameterReadout";

interface RecipeParameterListProps {
  recipe: Recipe;
}

export function RecipeParameterList({ recipe }: RecipeParameterListProps) {
  const { whiteBalance } = recipe;

  const whiteBalanceLabel =
    whiteBalance.mode === "Kelvin" && whiteBalance.kelvin ? `${whiteBalance.kelvin}K` : whiteBalance.mode;

  const hasCameraSettings =
    recipe.noiseReduction !== undefined || recipe.isoRange !== undefined || recipe.exposureCompensation !== undefined;

  return (
    <div className="space-y-4">
      <ParameterReadout label="Film Simulation" variant="badge" value={recipe.baseFilmSimulation} />
      <ParameterReadout label="Dynamic Range" variant="badge" value={recipe.dynamicRange} />
      <ParameterReadout label="White Balance" variant="badge" value={whiteBalanceLabel} />
      <ParameterReadout label="WB Shift — Red" variant="slider" value={whiteBalance.shift.red} min={-9} max={9} />
      <ParameterReadout label="WB Shift — Blue" variant="slider" value={whiteBalance.shift.blue} min={-9} max={9} />
      <ParameterReadout label="Highlight Tone" variant="slider" value={recipe.highlightTone} min={-2} max={4} />
      <ParameterReadout label="Shadow Tone" variant="slider" value={recipe.shadowTone} min={-2} max={4} />
      <ParameterReadout label="Color" variant="slider" value={recipe.color} min={-4} max={4} />
      <ParameterReadout label="Sharpness" variant="slider" value={recipe.sharpness} min={-4} max={4} />
      <ParameterReadout label="Color Chrome Effect" variant="segments" value={recipe.colorChromeEffect} />
      <ParameterReadout label="Color Chrome FX Blue" variant="segments" value={recipe.colorChromeFxBlue} />
      <ParameterReadout label="Grain Effect" variant="segments" value={recipe.grainEffect} />
      {recipe.grainSize && <ParameterReadout label="Grain Size" variant="badge" value={recipe.grainSize} />}

      {hasCameraSettings && (
        <div className="space-y-3 border-t border-ink-800 pt-4">
          <p className="text-xs font-bold uppercase tracking-wider text-ink-500">
            Recommended Camera Settings
          </p>
          <p className="text-[11px] text-ink-600">
            Dial these in on the camera before shooting — they affect capture, not something a preview can apply
            after the fact.
          </p>
          {recipe.isoRange !== undefined && <ParameterReadout label="ISO" variant="badge" value={recipe.isoRange} />}
          {recipe.exposureCompensation !== undefined && (
            <ParameterReadout
              label="Exposure Compensation"
              variant="badge"
              value={
                typeof recipe.exposureCompensation === "number"
                  ? `${recipe.exposureCompensation > 0 ? "+" : ""}${recipe.exposureCompensation} EV`
                  : recipe.exposureCompensation
              }
            />
          )}
          {recipe.noiseReduction !== undefined && (
            <ParameterReadout label="Noise Reduction" variant="slider" value={recipe.noiseReduction} min={-4} max={4} />
          )}
          {recipe.clarity !== undefined && (
            <ParameterReadout label="Clarity" variant="slider" value={recipe.clarity} min={-5} max={5} />
          )}
        </div>
      )}
    </div>
  );
}
