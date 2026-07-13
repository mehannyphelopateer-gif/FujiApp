import type { BaseFilmSimulation, EffectStrength, GrainSize, WhiteBalanceMode, WhiteBalanceShift } from "@/types/recipe";

/**
 * Camera settings read back out of a Fuji JPEG's MakerNotes — i.e. what the
 * camera actually baked into the pixels, as opposed to what a recipe wants.
 * See src/lib/recipes/neutralize.ts for how this is turned into a delta.
 */
export interface DetectedSettings {
  cameraModel: string | null;
  baseFilmSimulation: BaseFilmSimulation | "Unknown";
  whiteBalance: {
    mode: WhiteBalanceMode | "Unknown";
    shift: WhiteBalanceShift;
  };
  highlightTone: number;
  shadowTone: number;
  /** Saturation/Color, neutralized like highlight/shadow. */
  color: number;
  sharpness: number;
  /** Already baked into the pixels — used to avoid stacking a second helping on top. */
  colorChromeEffect: EffectStrength;
  colorChromeFxBlue: EffectStrength;
  grainEffect: EffectStrength;
  /** Only meaningful when grainEffect !== "Off". */
  grainSize?: GrainSize;
}
