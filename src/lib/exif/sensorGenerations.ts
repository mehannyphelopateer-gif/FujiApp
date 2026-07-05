/**
 * Camera model -> X-Trans (or Bayer) sensor generation, used to filter the
 * recipe list to what's physically compatible with the shooting body (see
 * Recipe.compatibleSensors in src/types/recipe.ts).
 *
 * Not exhaustive — covers the common X-series bodies. Unlisted/unrecognized
 * models fall back to null, and callers should show all recipes (fail open)
 * rather than hide everything because detection was inconclusive.
 */
const CAMERA_SENSOR_MAP: Record<string, string> = {
  "X-T1": "X-Trans II",
  "X-T2": "X-Trans III",
  "X-T3": "X-Trans IV",
  "X-T4": "X-Trans IV",
  "X-T5": "X-Trans V",
  "X-T50": "X-Trans V",
  "X-T30": "X-Trans III",
  "X-T30 II": "X-Trans IV",
  "X-T20": "X-Trans III",
  "X-T10": "X-Trans II",
  "X-Pro1": "X-Trans I",
  "X-Pro2": "X-Trans III",
  "X-Pro3": "X-Trans IV",
  "X-H1": "X-Trans III",
  "X-H2": "X-Trans V",
  "X-H2S": "X-Trans V",
  "X-S10": "X-Trans IV",
  "X-S20": "X-Trans V",
  "X-E1": "X-Trans I",
  "X-E2": "X-Trans II",
  "X-E3": "X-Trans III",
  "X-E4": "X-Trans IV",
  "X-E5": "X-Trans V",
  "X-M5": "X-Trans V",
  "X100": "Bayer",
  "X100S": "X-Trans I",
  "X100T": "X-Trans II",
  "X100F": "X-Trans III",
  "X100V": "X-Trans IV",
  "X100VI": "X-Trans V",
};

export function mapCameraModelToSensorGeneration(model: string | null): string | null {
  if (!model) return null;
  const normalized = model.trim();

  if (CAMERA_SENSOR_MAP[normalized]) return CAMERA_SENSOR_MAP[normalized];

  // EXIF Model strings sometimes include extra padding/prefixes (e.g. "X-T5 ").
  // Match the longest known key first so "X-T30 II" wins over "X-T30".
  const sortedKeys = Object.keys(CAMERA_SENSOR_MAP).sort((a, b) => b.length - a.length);
  const found = sortedKeys.find((key) => normalized.includes(key));
  return found ? CAMERA_SENSOR_MAP[found] : null;
}
