import { useRef, useState } from "react";
import { useCameraLink } from "@/context/CameraLinkContext";
import type { CalibrationRecipe } from "@/lib/camera/calibrationRecipes";
import { decodeNeutralRaf } from "@/lib/raw/rawService";
import { saveManyToFiles } from "@/lib/photo/shareFile";

interface CaptureResult {
  label: string;
  status: "ok" | "error";
  error?: string;
}

interface CalibrationCaptureProps {
  rafFile: File;
  recipes: CalibrationRecipe[];
  /**
   * Phase 1/2 (film-sim LUTs) needs a true RAW-decoded neutral base
   * (decodeNeutralRaf) as one side of every pixel-correspondence pair.
   * Phase 3 (white balance/tone/saturation/Color Chrome/grain) instead
   * pairs each real conversion against the ALREADY-captured
   * `calib-provia.jpg` from a Phase 1/2 shoot — those parameters transform
   * already-rendered pixels, not raw sensor data, so decodeNeutralRaf isn't
   * the right baseline and doesn't need re-decoding. Set true to skip it —
   * the resulting export then has no calib-neutral.jpg, so it MUST be
   * saved into a shoot folder that already has one from a prior Phase 1/2
   * run of this same RAF, or the derivation scripts have nothing to pair
   * against.
   */
  skipNeutralDecode?: boolean;
}

/**
 * One-time offline calibration tool, not a regular user-facing feature: for
 * each entry in `recipes`, converts the loaded RAF through the real camera
 * (same convertWithRecipe pipeline RecipeQaSweep drives), plus (unless
 * `skipNeutralDecode`) a neutral base from decodeNeutralRaf. Every converted
 * image is held in memory and exported together as
 * calib-neutral.jpg/calib-<slug>.jpg in one "Save to..." picker session at
 * the end (saveManyToFiles) — one dialog per file was impractically tedious
 * once the recipe lists grew past a handful of entries. Move the saved
 * folder into the relevant derivation script's input folder afterward —
 * see the plan doc for the full methodology.
 */
export function CalibrationCapture({ rafFile, recipes, skipNeutralDecode }: CalibrationCaptureProps) {
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

    if (!skipNeutralDecode) {
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
    }

    for (const { slug, recipe } of recipes) {
      if (cancelRef.current) break;
      setProgress(`Converting ${slug}…`);

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

  const totalFiles = recipes.length + (skipNeutralDecode ? 0 : 1);

  return (
    <div className="space-y-3">
      <p className="text-[11px] text-ink-500">
        {skipNeutralDecode
          ? `Exports a real camera conversion for each of ${recipes.length} test settings, for measuring the camera's real response offline (see the plan doc's Phase 3 section). Save these into the SAME shoot folder as an existing Phase 1/2 run of this RAF — the derivation scripts pair each file against that folder's calib-provia.jpg.`
          : `Exports a neutral base image plus a real camera conversion for each of ${recipes.length} film simulations, for deriving real Hald CLUTs offline (see the plan doc).`}{" "}
        All {totalFiles} files are saved together in one "Save to..." dialog at the end — pick a folder
        (e.g. a new "Shoot N" folder) and every file lands there. Move that folder into the derivation
        script's input folder afterward.
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
