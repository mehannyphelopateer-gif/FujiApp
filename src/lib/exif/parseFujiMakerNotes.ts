import ExifReader, { type ExpandedTags } from "exifreader";
import type { DetectedSettings } from "@/types/exif";
import type { BaseFilmSimulation, EffectStrength, GrainSize, WhiteBalanceMode, WhiteBalanceShift } from "@/types/recipe";

/**
 * Verified against a real Fuji MakerNote block (X-Pro1 sample RAF's embedded
 * preview JPEG, 2026-07) — earlier versions of this file assumed ExifReader
 * auto-parsed Fuji MakerNotes into a `tags.Fujifilm` group the way it does
 * for Canon/Pentax, which turned out to be untrue: ExifReader has no
 * Fujifilm-specific MakerNote support at all (confirmed by reading its own
 * source — only canon-tags.js and pentax-tags.js exist), so `tags.Fujifilm`
 * was always `undefined` and this function always returned null. It now
 * parses the raw `tags.exif.MakerNote` binary IFD directly.
 *
 * Tag IDs and their PrintConv tables below are transcribed from exiftool's
 * FujiFilm.pm (the authoritative, actively-maintained reverse-engineering of
 * this format — https://github.com/exiftool/exiftool), cross-checked against
 * the real sample file's decoded bytes.
 *
 * Fuji MakerNote structure: 8-byte "FUJIFILM" signature, then a little-endian
 * uint32 IFD offset (relative to the signature's start), then a standard
 * TIFF-style IFD at that offset: uint16 entry count, followed by 12-byte
 * entries (uint16 tag, uint16 type, uint32 count, 4-byte value-or-offset).
 * All the tags this function reads are single SHORT/SLONG values that fit
 * inline in that 4-byte field, except WhiteBalanceFineTune (2 x SLONG),
 * which is offset-based.
 */

const SIGNATURE = "FUJIFILM";
const TYPE_SHORT = 3;
const TYPE_SLONG = 9;

interface IfdEntry {
  tag: number;
  type: number;
  count: number;
  /** Byte offset within the MakerNote buffer of this entry's 4-byte value/offset field. */
  valueFieldOffset: number;
}

function parseFujiIfd(bytes: Uint8Array): Map<number, IfdEntry> | null {
  if (bytes.length < 12) return null;
  const signature = new TextDecoder("latin1").decode(bytes.slice(0, SIGNATURE.length));
  if (signature !== SIGNATURE) return null;

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const ifdOffset = view.getUint32(8, true);
  if (ifdOffset + 2 > bytes.length) return null;

  const numEntries = view.getUint16(ifdOffset, true);
  const entries = new Map<number, IfdEntry>();
  for (let i = 0; i < numEntries; i++) {
    const entryOffset = ifdOffset + 2 + i * 12;
    if (entryOffset + 12 > bytes.length) break;
    const tag = view.getUint16(entryOffset, true);
    const type = view.getUint16(entryOffset + 2, true);
    const count = view.getUint32(entryOffset + 4, true);
    entries.set(tag, { tag, type, count, valueFieldOffset: entryOffset + 8 });
  }
  return entries;
}

/** Reads an inline SHORT (uint16) value from an entry's value field. */
function readInlineShort(view: DataView, entry: IfdEntry): number | null {
  if (entry.type !== TYPE_SHORT || entry.count !== 1) return null;
  return view.getUint16(entry.valueFieldOffset, true);
}

/** Reads an inline SLONG (int32) value from an entry's value field. */
function readInlineSLong(view: DataView, entry: IfdEntry): number | null {
  if (entry.type !== TYPE_SLONG || entry.count !== 1) return null;
  return view.getInt32(entry.valueFieldOffset, true);
}

/** Reads an offset-based array of SLONG (int32) values (e.g. WhiteBalanceFineTune's Red/Blue pair). */
function readOffsetSLongArray(view: DataView, bytes: Uint8Array, entry: IfdEntry): number[] | null {
  if (entry.type !== TYPE_SLONG) return null;
  const dataOffset = view.getUint32(entry.valueFieldOffset, true);
  if (dataOffset + entry.count * 4 > bytes.length) return null;
  const values: number[] = [];
  for (let i = 0; i < entry.count; i++) {
    values.push(view.getInt32(dataOffset + i * 4, true));
  }
  return values;
}

// ---- 0x1401 FilmMode -> BaseFilmSimulation ----
// Per exiftool: "this doesn't seem to work for the X100" on some older
// bodies, where it reads 0 (Standard/Provia) regardless of the actual film
// simulation in use — SATURATION_BW_MAP below covers that case, since B&W
// simulations (Acros/Monochrome/Sepia) show up reliably in the Saturation tag.
const FILM_MODE_MAP: Record<number, BaseFilmSimulation> = {
  0x000: "Provia",
  0x120: "Astia",
  0x200: "Velvia",
  0x400: "Velvia",
  0x500: "Pro Neg Std",
  0x501: "Pro Neg Hi",
  0x600: "Classic Chrome",
  0x700: "Eterna",
  0x800: "Classic Negative",
  0x900: "Eterna Bleach Bypass",
  0xa00: "Nostalgic Neg",
  0xb00: "Reala Ace",
};

// ---- 0x1003 Saturation -> BaseFilmSimulation, for B&W variants FilmMode misses ----
const SATURATION_BW_MAP: Record<number, BaseFilmSimulation> = {
  0x300: "Monochrome",
  0x301: "Monochrome",
  0x302: "Monochrome",
  0x303: "Monochrome",
  0x310: "Sepia",
  0x500: "Acros",
  0x501: "Acros",
  0x502: "Acros",
  0x503: "Acros",
};

// ---- 0x1003 Saturation -> numeric Color (-4..+4), for non-B&W values ----
const SATURATION_COLOR_MAP: Record<number, number> = {
  0x000: 0,
  0x080: 1,
  0x0c0: 3,
  0x0e0: 4,
  0x100: 2,
  0x180: -1,
  0x200: -2,
  0x400: -2,
  0x4c0: -3,
  0x4e0: -4,
};

// ---- 0x1002 WhiteBalance -> WhiteBalanceMode ----
const WHITE_BALANCE_MAP: Record<number, WhiteBalanceMode> = {
  0x0: "Auto",
  0x1: "Auto",
  0x2: "Auto",
  0x100: "Daylight",
  0x200: "Shade",
  0x300: "Fluorescent1",
  0x301: "Fluorescent1",
  0x302: "Fluorescent2",
  0x303: "Fluorescent3",
  0x304: "Fluorescent3",
  0x400: "Incandescent",
  0x600: "Underwater",
  0xff0: "Kelvin",
};

// ---- 0x1047 GrainEffectRoughness / 0x1048 ColorChromeEffect / 0x104e ColorChromeFXBlue ----
// All three share the same 0/32/64 -> Off/Weak/Strong encoding, per exiftool's FujiFilm.pm.
const EFFECT_STRENGTH_MAP: Record<number, EffectStrength> = { 0: "Off", 32: "Weak", 64: "Strong" };

// ---- 0x104c GrainEffectSize -> Off/Small/Large ----
const GRAIN_SIZE_MAP: Record<number, GrainSize | "Off"> = { 0: "Off", 16: "Small", 32: "Large" };

// ---- 0x1001 Sharpness -> numeric (-4..+4) ----
// Not a linear scale — exiftool's own table jumps 0x02(-2) -> 0x03(0) ->
// 0x04(+2), with -1/+1 living at the unrelated-looking 0x82/0x84.
const SHARPNESS_MAP: Record<number, number> = {
  0x00: -4,
  0x01: -3,
  0x02: -2,
  0x03: 0,
  0x04: 2,
  0x05: 3,
  0x06: 4,
  0x82: -1,
  0x84: 1,
};

/** HighlightTone/ShadowTone: raw SLONG, displayed value = -raw/16. */
function decodeTone(raw: number | null): number {
  if (raw === null) return 0;
  return -raw / 16;
}

/** Fuji's WhiteBalanceFineTune is stored well outside the -9..+9 UI scale; /20 per exiftool's own tag notes. */
const WB_FINE_TUNE_SCALE = 20;

function extractMakerNoteBytes(makerNote: unknown): Uint8Array | null {
  if (!makerNote || typeof makerNote !== "object") return null;
  const raw = (makerNote as { value?: unknown }).value;
  if (!raw || typeof raw !== "object" || typeof (raw as ArrayLike<number>).length !== "number") return null;
  return new Uint8Array(raw as ArrayLike<number>);
}

/**
 * Extracts the camera's baked-in film simulation settings from a Fuji
 * JPEG's EXIF MakerNotes. Returns null if the file has no Fuji MakerNotes
 * (non-Fuji camera, or EXIF stripped by prior editing) — callers fall back
 * to a neutral baseline (see computeRecipeAdjustment in lib/recipes/neutralize.ts).
 */
export async function extractDetectedSettings(file: File): Promise<DetectedSettings | null> {
  let tags: ExpandedTags;
  try {
    tags = await ExifReader.load(file, { expanded: true });
  } catch {
    return null;
  }

  const makerNoteBytes = extractMakerNoteBytes(tags.exif?.MakerNote);
  if (!makerNoteBytes) return null;

  const entries = parseFujiIfd(makerNoteBytes);
  if (!entries) return null;

  const view = new DataView(makerNoteBytes.buffer, makerNoteBytes.byteOffset, makerNoteBytes.byteLength);

  const filmModeEntry = entries.get(0x1401);
  const filmModeRaw = filmModeEntry ? readInlineShort(view, filmModeEntry) : null;
  const saturationEntry = entries.get(0x1003);
  const saturationRaw = saturationEntry ? readInlineShort(view, saturationEntry) : null;

  const baseFilmSimulation: BaseFilmSimulation | "Unknown" =
    (saturationRaw !== null ? SATURATION_BW_MAP[saturationRaw] : undefined) ??
    (filmModeRaw !== null ? FILM_MODE_MAP[filmModeRaw] : undefined) ??
    "Unknown";

  const wbEntry = entries.get(0x1002);
  const wbRaw = wbEntry ? readInlineShort(view, wbEntry) : null;
  const wbMode: WhiteBalanceMode | "Unknown" = (wbRaw !== null ? WHITE_BALANCE_MAP[wbRaw] : undefined) ?? "Unknown";

  const wbFineTuneEntry = entries.get(0x100a);
  const wbFineTuneRaw = wbFineTuneEntry ? readOffsetSLongArray(view, makerNoteBytes, wbFineTuneEntry) : null;
  const shift: WhiteBalanceShift =
    wbFineTuneRaw && wbFineTuneRaw.length >= 2
      ? { red: wbFineTuneRaw[0] / WB_FINE_TUNE_SCALE, blue: wbFineTuneRaw[1] / WB_FINE_TUNE_SCALE }
      : { red: 0, blue: 0 };

  const highlightEntry = entries.get(0x1041);
  const highlightTone = decodeTone(highlightEntry ? readInlineSLong(view, highlightEntry) : null);
  const shadowEntry = entries.get(0x1040);
  const shadowTone = decodeTone(shadowEntry ? readInlineSLong(view, shadowEntry) : null);

  const color =
    saturationRaw !== null && SATURATION_COLOR_MAP[saturationRaw] !== undefined ? SATURATION_COLOR_MAP[saturationRaw] : 0;

  const sharpnessEntry = entries.get(0x1001);
  const sharpnessRaw = sharpnessEntry ? readInlineShort(view, sharpnessEntry) : null;
  const sharpness = sharpnessRaw !== null && SHARPNESS_MAP[sharpnessRaw] !== undefined ? SHARPNESS_MAP[sharpnessRaw] : 0;

  const grainRoughnessEntry = entries.get(0x1047);
  const grainRoughnessRaw = grainRoughnessEntry ? readInlineShort(view, grainRoughnessEntry) : null;
  const grainEffect: EffectStrength =
    (grainRoughnessRaw !== null ? EFFECT_STRENGTH_MAP[grainRoughnessRaw] : undefined) ?? "Off";

  const grainSizeEntry = entries.get(0x104c);
  const grainSizeRaw = grainSizeEntry ? readInlineShort(view, grainSizeEntry) : null;
  const grainSizeDecoded = grainSizeRaw !== null ? GRAIN_SIZE_MAP[grainSizeRaw] : undefined;
  const grainSize: GrainSize | undefined = grainSizeDecoded && grainSizeDecoded !== "Off" ? grainSizeDecoded : undefined;

  const colorChromeEntry = entries.get(0x1048);
  const colorChromeRaw = colorChromeEntry ? readInlineShort(view, colorChromeEntry) : null;
  const colorChromeEffect: EffectStrength =
    (colorChromeRaw !== null ? EFFECT_STRENGTH_MAP[colorChromeRaw] : undefined) ?? "Off";

  const colorChromeFxBlueEntry = entries.get(0x104e);
  const colorChromeFxBlueRaw = colorChromeFxBlueEntry ? readInlineShort(view, colorChromeFxBlueEntry) : null;
  const colorChromeFxBlue: EffectStrength =
    (colorChromeFxBlueRaw !== null ? EFFECT_STRENGTH_MAP[colorChromeFxBlueRaw] : undefined) ?? "Off";

  return {
    cameraModel: tags.exif?.Model?.description ?? null,
    baseFilmSimulation,
    whiteBalance: { mode: wbMode, shift },
    highlightTone,
    shadowTone,
    color,
    sharpness,
    colorChromeEffect,
    colorChromeFxBlue,
    grainEffect,
    grainSize,
  };
}
