import type { Recipe } from "@/types/recipe";
import {
  DR_ENCODE,
  EFFECT_ENCODE,
  FILM_SIM_ENCODE,
  MONOCHROME_SIMS,
  NR_ENCODE,
  WB_MODE_ENCODE,
  encodeGrain,
  tone,
} from "@/lib/camera/encodeRecipe";

/**
 * Patches the camera's RAW-conversion profile (property 0xD185, read via
 * CameraLink.getRawProfile) with a Recipe's settings, so triggering
 * conversion (0xD183) produces a real, camera-rendered JPEG for that recipe
 * — not just the 7 recipes saved to C1-C7 custom slots.
 *
 * Confirmed against real hardware (2026-07): the profile is a ~625-byte
 * blob. The first 2 bytes are `numParams` (UInt16 LE) — the count of
 * trailing Int32 LE parameters. The parameter array starts at
 * `profileBytes.length - numParams * 4`; a field's value lives at
 * `off + NativeIdx * 4`. Everything before that offset (image dimensions,
 * a device/serial string, etc. — none of it relevant here) is left
 * completely untouched, matching filmkit's own patchProfile design
 * philosophy: only overwrite fields this app actually knows the meaning
 * of, so the camera falls back to its own sensible/EXIF defaults for
 * everything else (this is also why the profile must always be read fresh
 * per-file rather than cached/reused — the sentinel-preserving unpatched
 * fields differ per shot).
 *
 * Value encodings are identical to the C1-C7 preset property encodings in
 * encodeRecipe.ts (same tables, imported directly) — only the addressing
 * differs (a field index into one big property blob, vs. a separate PTP
 * property per field).
 */
export const NATIVE_IDX = {
  exposureBias: 4,
  dynamicRange: 6,
  wideDRange: 7,
  filmSimulation: 8,
  grainEffect: 9,
  colorChrome: 10,
  smoothSkin: 11,
  whiteBalance: 12,
  wbShiftR: 13,
  wbShiftB: 14,
  wbColorTemp: 15,
  highlightTone: 16,
  shadowTone: 17,
  color: 18,
  sharpness: 19,
  noiseReduction: 20,
  ccFxBlue: 25,
  clarity: 27,
} as const;

export interface PatchRawProfileOptions {
  /**
   * WhiteBalance's raw encoding inside this profile (as opposed to the C1-C7
   * preset property) is unconfirmed by any reference source — 0 is a
   * documented sentinel meaning "use the shot's as-shot WB," which is what
   * happens by default (left untouched) unless this is explicitly set.
   */
  forceWhiteBalance?: boolean;
}

/**
 * Returns a *new* Uint8Array — never mutates the input, since callers
 * typically want to re-patch the same freshly-read base profile for
 * multiple different recipes without it accumulating stale values.
 */
export function patchRawProfile(profileBytes: Uint8Array, recipe: Recipe, options: PatchRawProfileOptions = {}): Uint8Array {
  const patched = new Uint8Array(profileBytes);
  const view = new DataView(patched.buffer, patched.byteOffset, patched.byteLength);

  const numParams = view.getUint16(0, true);
  const paramsOffset = patched.length - numParams * 4;

  function setParam(idx: number, value: number) {
    view.setInt32(paramsOffset + idx * 4, value, true);
  }

  const isMono = MONOCHROME_SIMS.has(recipe.baseFilmSimulation);

  setParam(NATIVE_IDX.dynamicRange, DR_ENCODE[recipe.dynamicRange]);
  setParam(NATIVE_IDX.filmSimulation, FILM_SIM_ENCODE[recipe.baseFilmSimulation] ?? 0x01);
  setParam(NATIVE_IDX.grainEffect, encodeGrain(recipe.grainEffect, recipe.grainSize));
  setParam(NATIVE_IDX.colorChrome, EFFECT_ENCODE[recipe.colorChromeEffect]);
  setParam(NATIVE_IDX.ccFxBlue, EFFECT_ENCODE[recipe.colorChromeFxBlue]);
  setParam(NATIVE_IDX.highlightTone, tone(recipe.highlightTone));
  setParam(NATIVE_IDX.shadowTone, tone(recipe.shadowTone));
  if (!isMono) {
    setParam(NATIVE_IDX.color, tone(recipe.color));
  }
  setParam(NATIVE_IDX.sharpness, tone(recipe.sharpness));
  setParam(NATIVE_IDX.wbShiftR, recipe.whiteBalance.shift.red);
  setParam(NATIVE_IDX.wbShiftB, recipe.whiteBalance.shift.blue);

  if (recipe.noiseReduction !== undefined) {
    const clamped = Math.max(-4, Math.min(4, Math.round(recipe.noiseReduction)));
    setParam(NATIVE_IDX.noiseReduction, NR_ENCODE[clamped]);
  }
  if (recipe.clarity !== undefined) {
    setParam(NATIVE_IDX.clarity, tone(recipe.clarity));
  }

  if (options.forceWhiteBalance) {
    setParam(NATIVE_IDX.whiteBalance, WB_MODE_ENCODE[recipe.whiteBalance.mode] ?? WB_MODE_ENCODE.Auto);
    if (recipe.whiteBalance.mode === "Kelvin" && recipe.whiteBalance.kelvin) {
      setParam(NATIVE_IDX.wbColorTemp, recipe.whiteBalance.kelvin);
    }
  }

  // exposureBias/wideDRange/smoothSkin and every index outside NATIVE_IDX are
  // intentionally left exactly as read — no Recipe field cleanly maps to
  // them, and preserving the camera's own sentinel/default keeps conversion
  // from rejecting the write or producing a nonsensical result.

  return patched;
}
