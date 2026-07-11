import type { BaseFilmSimulation, EffectStrength, GrainSize, Recipe } from "@/types/recipe";

/**
 * Encodes a Recipe into the ordered {id, value} property list the camera
 * expects for a Custom Setting slot write. Inverse of decodeSlot.ts, and the
 * write-side counterpart of the same tables ported from filmkit's
 * src/profile/preset-translate.ts (translateUIToPresetProps) — including its
 * write ORDER, which matters: D19C must immediately follow D199, and D19A/
 * D19B follow that, matching what filmkit found the official app does.
 *
 * Two Recipe fields the shader treats as "informational only" — noiseReduction
 * and clarity — are still written here: that restriction was about the
 * *preview shader* not being able to simulate in-camera noise reduction or
 * clarity after the fact, not about whether the camera's own Custom Setting
 * banks store them. D1A1/D1A2 are real, writable preset properties. ISO range
 * and exposure compensation genuinely aren't part of the preset property set
 * (filmkit's own translateUIToPresetProps excludes them too) and stay excluded.
 */

const FILM_SIM_ENCODE: Record<BaseFilmSimulation, number> = {
  Provia: 0x01,
  Velvia: 0x02,
  Astia: 0x03,
  "Pro Neg Hi": 0x04,
  "Pro Neg Std": 0x05,
  Monochrome: 0x06,
  Sepia: 0x0a,
  "Classic Chrome": 0x0b,
  Acros: 0x0c,
  Eterna: 0x10,
  "Classic Negative": 0x11,
  "Eterna Bleach Bypass": 0x12,
  "Nostalgic Neg": 0x13,
  "Reala Ace": 0x14,
};

const MONOCHROME_SIMS = new Set<BaseFilmSimulation>(["Monochrome", "Sepia", "Acros"]);

const WB_MODE_ENCODE: Record<Recipe["whiteBalance"]["mode"], number> = {
  Auto: 0x0002,
  Daylight: 0x0004,
  Incandescent: 0x0006,
  Underwater: 0x0008,
  Fluorescent1: 0x8001,
  Fluorescent2: 0x8002,
  Fluorescent3: 0x8003,
  Shade: 0x8006,
  Kelvin: 0x8007,
};

const EFFECT_ENCODE: Record<EffectStrength, number> = { Off: 1, Weak: 2, Strong: 3 };

/** Preset grain is a flat 1-5 enum (strength x size combined). Missing
 *  grainSize when grainEffect isn't Off defaults to Small. */
function encodeGrain(effect: EffectStrength, size: GrainSize | undefined): number {
  if (effect === "Off") return 1;
  const isLarge = size === "Large";
  if (effect === "Weak") return isLarge ? 4 : 2;
  return isLarge ? 5 : 3; // Strong
}

/** Fuji's proprietary HighIsoNR encoding (NOT x10, NOT linear). Ported from filmkit's NR_ENCODE. */
const NR_ENCODE: Record<number, number> = {
  [-4]: 0x8000,
  [-3]: 0x7000,
  [-2]: 0x4000,
  [-1]: 0x3000,
  [0]: 0x2000,
  [1]: 0x1000,
  [2]: 0x0000,
  [3]: 0x6000,
  [4]: 0x5000,
};

/** Observed-safe defaults for fields the Recipe model doesn't track at all. */
const UNKNOWN_DEFAULTS = {
  imageSize: 7, // L 3:2
  imageQuality: 4,
  d191: 0,
  smoothSkin: 1, // Off (1-indexed)
  longExpNR: 1, // On
  colorSpace: 1, // sRGB
  d1a5: 7,
};

const tone = (value: number) => Math.round(value * 10);

export interface EncodedProperty {
  id: string; // hex, e.g. "D19D" — matches decodeSlot.ts's key format
  value: number;
}

/**
 * The camera rejected a real-world preset name ("Cito's porta 800", which
 * contains U+2019 — a curly/typographic apostrophe, not the plain ASCII one)
 * with PTP's "Invalid Device Prop Value" — the on-camera preset name field
 * almost certainly only accepts basic ASCII. Transliterates common
 * typographic punctuation to its ASCII equivalent, then strips anything
 * still outside printable ASCII (e.g. accented letters without a
 * decomposition, like Polish "ł") rather than sending bytes the camera may
 * reject entirely.
 */
export function sanitizeCameraName(name: string): string {
  const asciiPunctuation = name
    .replace(/[‘’]/g, "'") // left/right single quotation mark
    .replace(/[“”]/g, '"') // left/right double quotation mark
    .replace(/[–—]/g, "-") // en/em dash
    .replace(/…/g, "...");
  // Strips combining diacritical marks left behind by NFKD decomposition
  // (e.g. "é" -> "e" + U+0301 combining acute accent).
  const withoutDiacritics = asciiPunctuation.normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
  return withoutDiacritics.replace(/[^\x20-\x7e]/g, "");
}

export function encodeRecipe(recipe: Recipe): EncodedProperty[] {
  const isMono = MONOCHROME_SIMS.has(recipe.baseFilmSimulation);
  const props: EncodedProperty[] = [];
  const push = (id: string, value: number) => props.push({ id, value });

  push("D18E", UNKNOWN_DEFAULTS.imageSize);
  push("D18F", UNKNOWN_DEFAULTS.imageQuality);
  push("D190", { DR100: 100, DR200: 200, DR400: 400, "DR-AUTO": 100 }[recipe.dynamicRange]);
  push("D191", UNKNOWN_DEFAULTS.d191);
  push("D192", FILM_SIM_ENCODE[recipe.baseFilmSimulation] ?? 0x01);

  push("D195", encodeGrain(recipe.grainEffect, recipe.grainSize));
  push("D196", EFFECT_ENCODE[recipe.colorChromeEffect]);
  push("D197", EFFECT_ENCODE[recipe.colorChromeFxBlue]);
  push("D198", UNKNOWN_DEFAULTS.smoothSkin);

  const wbMode = WB_MODE_ENCODE[recipe.whiteBalance.mode] ?? WB_MODE_ENCODE.Auto;
  push("D199", wbMode);
  // Must immediately follow D199 — the camera rejects it otherwise.
  if (recipe.whiteBalance.mode === "Kelvin" && recipe.whiteBalance.kelvin) {
    push("D19C", recipe.whiteBalance.kelvin);
  }
  push("D19A", recipe.whiteBalance.shift.red);
  push("D19B", recipe.whiteBalance.shift.blue);

  push("D19D", tone(recipe.highlightTone));
  push("D19E", tone(recipe.shadowTone));
  if (!isMono) {
    push("D19F", tone(recipe.color));
  }
  push("D1A0", tone(recipe.sharpness));

  if (recipe.noiseReduction !== undefined) {
    const clamped = Math.max(-4, Math.min(4, Math.round(recipe.noiseReduction)));
    push("D1A1", NR_ENCODE[clamped]);
  }
  push("D1A2", tone(recipe.clarity ?? 0));

  push("D1A3", UNKNOWN_DEFAULTS.longExpNR);
  push("D1A4", UNKNOWN_DEFAULTS.colorSpace);
  push("D1A5", UNKNOWN_DEFAULTS.d1a5);

  return props;
}
