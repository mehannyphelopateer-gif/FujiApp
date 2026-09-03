import { useRef, useState } from "react";
import { useCameraLink } from "@/context/CameraLinkContext";
import { CALIBRATION_RECIPES } from "@/lib/camera/calibrationRecipes";
import { decodeNeutralRaf } from "@/lib/raw/rawService";
import { saveManyToFiles } from "@/lib/photo/shareFile";

interface CaptureResult {
  label: string;
  status: "ok" | "error";
  error?: string;
}

interface CalibrationCaptureProps {
  rafFile: File;
}

/**
 * One-time offline calibration tool, not a regular user-facing feature: for
 * each CALIBRATION_RECIPES entry, converts the loaded RAF through the real
 * camera (same convertWithRecipe pipeline RecipeQaSweep drives), plus a
 * neutral base from decodeNeutralRaf. Every converted image is held in
 * memory and exported together as calib-neutral.jpg/calib-<slug>.jpg in one
 * "Save to..." picker session at the end (saveManyToFiles) — with 14 film
 * sims that's 15 files; one dialog per file was impractically tedious once
 * Phase 2 added the remaining 11 sims to CALIBRATION_RECIPES. Move the saved
 * folder into scripts/derive-luts-from-calibration.mjs's input folder
 * afterward — see the plan doc for the full methodology.
 */
export function CalibrationCapture({ rafFile }: CalibrationCaptureProps) {
  const { convertWithRecipe } = useCameraLink();
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [results, setResults] = useState<CaptureResult[]>([]);
  const cancelRef = useRef(false);

  async function handleRun() {
    setIsRunning(true);
    setResults([]);
    cancelRef.current = false;

    const toSave: { blob: Blob; filename: string }[] = [];

    setProgress("Decoding neutral base image…");
    try {
      const neutralBlob = await decodeNeutralRaf(rafFile);
      if (!neutralBlob) throw new Error("Neutral RAW decode returned nothing (native iOS only).");
      toSave.push({ blob: neutralBlob, filename: "calib-neutral.jpg" });
      setResults((prev) => [...prev, { label: "neutral", status: "ok" }]);
    } catch (err) {
      setResults((prev) => [
        ...prev,
        { label: "neutral", status: "error", error: err instanceof Error ? err.message : "Decode failed." },
      ]);
    }

    for (const { slug, recipe } of CALIBRATION_RECIPES) {
      if (cancelRef.current) break;
      setProgress(`Converting ${recipe.baseFilmSimulation}…`);

      const outcome = await convertWithRecipe(recipe, rafFile);
      if (outcome.ok && outcome.imageUrl) {
        try {
          const blob = await (await fetch(outcome.imageUrl)).blob();
          toSave.push({ blob, filename: `calib-${slug}.jpg` });
          setResults((prev) => [...prev, { label: slug, status: "ok" }]);
        } catch (err) {
          setResults((prev) => [
            ...prev,
            { label: slug, status: "error", error: err instanceof Error ? err.message : "Export failed." },
          ]);
        }
      } else {
        setResults((prev) => [...prev, { label: slug, status: "error", error: outcome.error }]);
      }
    }

    if (toSave.length > 0) {
      setProgress(`Saving ${toSave.length} file(s)…`);
      try {
        await saveManyToFiles(toSave);
      } catch (err) {
        setResults((prev) => [
          ...prev,
          { label: "save", status: "error", error: err instanceof Error ? err.message : "Save failed." },
        ]);
      }
    }

    setIsRunning(false);
    setProgress(null);
  }

  return (
    <div className="space-y-3">
      <p className="text-[11px] text-ink-500">
        Exports a neutral base image plus a real camera conversion for each of {CALIBRATION_RECIPES.length}{" "}
        film simulations, for deriving real Hald CLUTs offline (see the plan doc). All{" "}
        {CALIBRATION_RECIPES.length + 1} files are saved together in one "Save to..." dialog at the end —
        pick a folder (e.g. a new "Shoot N" folder) and every file lands there. Move that folder into the
        derivation script's input folder afterward.
      </p>

      <button
        type="button"
        onClick={handleRun}
        disabled={isRunning}
        className="rounded-md bg-gold-500 px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-950 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {isRunning ? "Capturing…" : "Run Calibration Capture"}
      </button>

      {progress && (
        <p className="text-[11px] font-bold uppercase tracking-wide text-ink-400">{progress}</p>
      )}

      {results.length > 0 && (
        <div className="space-y-1">
          {results.map((result) => (
            <p
              key={result.label}
              className={`text-[11px] ${result.status === "ok" ? "text-green-400" : "text-red-400"}`}
            >
              {result.label}: {result.status === "ok" ? "saved" : result.error}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
