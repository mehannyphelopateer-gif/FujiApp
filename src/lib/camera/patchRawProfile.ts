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
   * 0 is a confirmed sentinel (matches filmkit's own reference source)
   * meaning "use the shot's as-shot WB," which is what happens by default
   * (left untouched) unless this is explicitly set. Even when set, real
   * hardware testing found white balance doesn't reliably override on a
   * source RAF shot with a dialed-in Kelvin temperature — under active
   * investigation whether that's specific to Kelvin-sourced RAFs or a
   * broader limitation of this RAW-conversion mechanism.
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
    // Confirmed against real hardware that neither Auto (0x0002, a value
    // distinct from the profile's actual "leave as shot" sentinel of
    // 0x0000) nor a concrete preset like Daylight (0x0004) reliably override
    // a source RAF's dialed-in Kelvin white balance: Auto reads back clean
    // but gets ignored by the conversion engine itself (converted EXIF still
    // reports the original Kelvin WB); Daylight doesn't even survive the
    // write (reads back as 0/AsShot). Currently under investigation whether
    // this is specific to Kelvin-sourced RAFs — see patchRawProfile's
    // options doc comment.
    setParam(NATIVE_IDX.whiteBalance, WB_MODE_ENCODE[recipe.whiteBalance.mode] ?? WB_MODE_ENCODE.Auto);
    // Always write wbColorTemp explicitly (to the recipe's Kelvin value, or
    // 0 otherwise) rather than only when the recipe's mode is Kelvin — if
    // the source RAF was shot (or a prior patch left the profile) with a
    // Kelvin WB, leaving this field untouched for a non-Kelvin recipe let
    // that stale color temp survive the mode flip to Auto/whatever, which
    // is a real, confirmed cause of the converted recipe still carrying the
    // source shot's original WB signature underneath.
    setParam(
      NATIVE_IDX.wbColorTemp,
      recipe.whiteBalance.mode === "Kelvin" ? (recipe.whiteBalance.kelvin ?? 0) : 0,
    );
  }

  // exposureBias/wideDRange/smoothSkin and every index outside NATIVE_IDX are
  // intentionally left exactly as read — no Recipe field cleanly maps to
  // them, and preserving the camera's own sentinel/default keeps conversion
  // from rejecting the write or producing a nonsensical result.

  return patched;
}

/**
 * Diffs every field this app actually patches (NATIVE_IDX) between the
 * bytes intended to be written and a follow-up profile read — catches a
 * write the camera silently rejected or reverted (confirmed real behavior:
 * the whiteBalance field in particular has been observed reading back as
 * whatever it already was, not what was just written) before it's
 * discovered only by eyeballing the converted photo or its EXIF afterward.
 */
export function diffPatchedFields(intended: Uint8Array, actual: Uint8Array): string[] {
  if (intended.length !== actual.length) {
    return [`Length mismatch: intended ${intended.length} bytes, read back ${actual.length} bytes.`];
  }
  const intendedView = new DataView(intended.buffer, intended.byteOffset, intended.byteLength);
  const actualView = new DataView(actual.buffer, actual.byteOffset, actual.byteLength);
  const numParams = intendedView.getUint16(0, true);
  const off = intended.length - numParams * 4;

  const diffs: string[] = [];
  for (const [name, idx] of Object.entries(NATIVE_IDX)) {
    const expected = intendedView.getInt32(off + idx * 4, true);
    const got = actualView.getInt32(off + idx * 4, true);
    if (expected !== got) {
      diffs.push(`${name}: wrote ${expected}, read back ${got}`);
    }
  }
  return diffs;
}
