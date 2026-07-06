export type BaseFilmSimulation =
  | "Provia"
  | "Velvia"
  | "Astia"
  | "Classic Chrome"
  | "Pro Neg Hi"
  | "Pro Neg Std"
  | "Classic Negative"
  | "Eterna"
  | "Eterna Bleach Bypass"
  | "Acros"
  | "Monochrome"
  | "Sepia"
  | "Nostalgic Neg"
  | "Reala Ace";

export type DynamicRange = "DR100" | "DR200" | "DR400" | "DR-AUTO";

export type WhiteBalanceMode =
  | "Auto"
  | "Daylight"
  | "Shade"
  | "Fluorescent1"
  | "Fluorescent2"
  | "Fluorescent3"
  | "Incandescent"
  | "Underwater"
  | "Kelvin";

export type EffectStrength = "Off" | "Weak" | "Strong";

export type GrainSize = "Small" | "Large";

export interface WhiteBalanceShift {
  /** Fuji Red/Blue WB shift axis, roughly -9..+9 */
  red: number;
  blue: number;
}

export interface WhiteBalanceSetting {
  mode: WhiteBalanceMode;
  /** Present only when mode === "Kelvin". */
  kelvin?: number;
  shift: WhiteBalanceShift;
}

export interface Recipe {
  id: string;
  name: string;
  description?: string;
  baseFilmSimulation: BaseFilmSimulation;
  dynamicRange: DynamicRange;
  whiteBalance: WhiteBalanceSetting;
  /** Fuji scale, -2 (soft) .. +4 (hard), 0.5 increments. */
  highlightTone: number;
  /** Fuji scale, -2 (soft) .. +4 (hard), 0.5 increments. */
  shadowTone: number;
  /** Fuji scale, -4 (muted) .. +4 (vivid), Saturation/Color. Neutralized like highlight/shadow — it's another baked-in tone characteristic. */
  color: number;
  /**
   * Fuji scale, -4 (soft) .. +4 (hard). Applied forward-only by the shader
   * — it targets this absolute value rather than being neutralized against
   * the image's detected in-camera sharpness, since reversing baked-in
   * sharpening in a real-time shader produces artifacts.
   */
  sharpness: number;
  colorChromeEffect: EffectStrength;
  /** X-Trans IV+ only. Blue-weighted variant of Color Chrome Effect. */
  colorChromeFxBlue: EffectStrength;
  grainEffect: EffectStrength;
  /** Only meaningful when grainEffect !== "Off". X-Trans IV+ only. */
  grainSize?: GrainSize;
  /**
   * Informational only — these are camera capture-time settings (what to
   * dial in before shooting), not something a post-process shader can
   * apply to an already-rendered JPEG. Displayed as "recommended camera
   * settings" in the UI; intentionally excluded from RecipeAdjustment.
   */
  noiseReduction?: number; // Fuji scale, -4..+4
  isoRange?: string; // e.g. "Auto, up to ISO 6400"
  /**
   * Recommended EV offset. A single value (e.g. 0.5 for +1/3 to +2/3 stop) or
   * a free-form range/condition string (e.g. "0 to +1") when the source
   * recipe gives a range rather than one number.
   */
  exposureCompensation?: number | string;
  /** Fuji scale, -5 (soft) .. +5 (hard). Informational only, same reasoning as noiseReduction above. */
  clarity?: number;
  /**
   * Forward-looking field for Phase 4: sensor generations this recipe is
   * considered compatible with. Filtering logic is built in Phase 4;
   * Phase 1 only needs the data present.
   */
  compatibleSensors: string[];
  /** True for user-created/edited recipes stored in localStorage, as opposed to the shipped catalog. */
  isCustom?: boolean;
  /**
   * Data-URL thumbnail captured from the live WebGL canvas right after the
   * user saves a custom recipe (while they have a photo loaded) — custom
   * recipes have no place in the pre-rendered /recipe-previews/ build step,
   * so this is the only way to show an accurate preview for one.
   */
  previewImage?: string;
}
