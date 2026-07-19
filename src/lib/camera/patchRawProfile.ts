import type { Recipe } from "@/types/recipe";
import {
  DR_ENCODE,
  EFFECT_ENCODE,
  FILM_SIM_ENCODE,
  MONOCHROME_SIMS,
  NR_ENCODE,
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

/**
 * Returns a *new* Uint8Array — never mutates the input, since callers
 * typically want to re-patch the same freshly-read base profile for
 * multiple different recipes without it accumulating stale values.
 */
export function patchRawProfile(profileBytes: Uint8Array, recipe: Recipe): Uint8Array {
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

  // whiteBalance (idx 12) and wbColorTemp (idx 15) are deliberately left
  // untouched — always as-shot. Confirmed against real hardware, across
  // three different source photos: Auto (0x0002, a value distinct from the
  // profile's actual "leave as shot" sentinel of 0x0000) reads back clean
  // but gets ignored by the conversion engine itself; a concrete preset
  // like Daylight (0x0004) doesn't even survive the write (reads back as
  // 0/AsShot); and forcing WB on a non-Kelvin-sourced RAF produced a teal
  // color cast (Classic Cuban Neg's Strong Color Chrome FX Blue amplifying
  // whatever the actual, not-quite-matching white balance turned out to
  // be). Leaving these untouched is reliable and predictable — the
  // WB *shift* (wbShiftR/wbShiftB) below still applies on top of whatever
  // as-shot WB the camera keeps.
  //
  // exposureBias/wideDRange/smoothSkin and every other index outside
  // NATIVE_IDX are likewise left exactly as read — no Recipe field cleanly
  // maps to them, and preserving the camera's own sentinel/default keeps
  // conversion from rejecting the write or producing a nonsensical result.

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
