import { useAppState } from "@/context/AppStateContext";
import { ParameterReadout } from "@/components/recipes/ParameterReadout";

export function DetectedSettingsPanel() {
  const { detectedSettings, selectedFile } = useAppState();

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
