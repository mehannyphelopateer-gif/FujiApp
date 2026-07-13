import { useMemo } from "react";
import { useAppState } from "@/context/AppStateContext";
import { ParameterReadout } from "@/components/recipes/ParameterReadout";
import { findMatchingRecipes } from "@/lib/recipes/matchRecipe";

export function DetectedSettingsPanel() {
  const { detectedSettings, selectedFile, recipes } = useAppState();

  // Matches against the full catalog, not `compatibleRecipes` (sensor-filtered
  // for the *currently uploaded* camera body) — a recipe's `compatibleSensors`
  // reflects which body it was designed/tested on, not a hard technical
  // restriction on which bodies can shoot it. Filtering the identification
  // pool by sensor excluded the actual recipe used whenever it was shot on a
  // body its listing doesn't mention (e.g. "Summer Chrome" is only tagged
  // X-Trans IV; shooting it on an X-Trans V body meant it never made the
  // candidate pool, so a near-identical but wrong recipe won instead).
  const topMatch = useMemo(() => {
    if (!detectedSettings) return null;
    return findMatchingRecipes(detectedSettings, recipes, 1)[0] ?? null;
  }, [detectedSettings, recipes]);

  if (!selectedFile) return null;

  if (!detectedSettings) {
    return (
      <p className="rounded-md border border-ink-800 bg-ink-900 px-3 py-2.5 text-xs text-ink-500">
        No Fuji metadata detected in this image — recipes will be applied directly.
      </p>
    );
  }

  const { whiteBalance } = detectedSettings;

  return (
    <div className="space-y-4 rounded-md border border-ink-800 bg-ink-900 p-3.5">
      {topMatch && (
        <div className="border-b border-ink-800 pb-4">
          <p className="text-[10px] font-bold uppercase tracking-wide text-ink-500">Recipe Used</p>
          <p className="text-base font-black leading-tight text-gold-400">{topMatch.recipe.name}</p>
          <p className="mt-0.5 text-[11px] text-ink-500">{topMatch.score}% match</p>
        </div>
      )}

      {detectedSettings.cameraModel && (
        <ParameterReadout label="Camera" variant="badge" value={detectedSettings.cameraModel} />
      )}
      <ParameterReadout
        label="Detected Film Simulation"
        variant="badge"
        value={detectedSettings.baseFilmSimulation}
      />
      <ParameterReadout label="Detected White Balance" variant="badge" value={whiteBalance.mode} />
      <ParameterReadout
        label="Detected Highlight Tone"
        variant="slider"
        value={detectedSettings.highlightTone}
        min={-2}
        max={4}
      />
      <ParameterReadout
        label="Detected Shadow Tone"
        variant="slider"
        value={detectedSettings.shadowTone}
        min={-2}
        max={4}
      />
      <ParameterReadout
        label="Detected Color"
        variant="slider"
        value={detectedSettings.color}
        min={-4}
        max={4}
      />
      <ParameterReadout
        label="Detected Sharpness"
        variant="slider"
        value={detectedSettings.sharpness}
        min={-4}
        max={4}
      />
    </div>
  );
}
