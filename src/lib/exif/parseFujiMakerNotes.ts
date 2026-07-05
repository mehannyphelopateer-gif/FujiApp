import ExifReader from "exifreader";
import type { DetectedSettings } from "@/types/exif";
import type { BaseFilmSimulation, WhiteBalanceMode, WhiteBalanceShift } from "@/types/recipe";

/**
 * IMPORTANT — unverified against real files. This app was built without a
 * runtime available to test with (no Node.js in the dev environment), so
 * the tag names and scaling constants below are best-effort, not confirmed.
 * Before relying on this: run `ExifReader.load(file, { expanded: true })`
 * against a handful of real Fuji JPEGs (ideally from different bodies /
 * firmware), log the raw `tags.Fujifilm` object, and correct anything below
 * that doesn't match. See src/lib/exif/README.md.
 */

interface FujiMakerNoteTags {
  FilmMode?: { description?: string };
  Saturation?: { description?: string; value?: number };
  Color?: { value?: number };
  WhiteBalance?: { description?: string };
  WhiteBalanceFineTune?: { value?: number[] | number };
  HighlightTone?: { value?: number };
  Highlight?: { value?: number };
  ShadowTone?: { value?: number };
  Shadow?: { value?: number };
  Sharpness?: { value?: number };
}

interface ExpandedExifTags {
  exif?: { Model?: { description?: string } };
  Fujifilm?: FujiMakerNoteTags;
}

const FILM_MODE_KEYWORDS: Array<[string, BaseFilmSimulation]> = [
  ["classic chrome", "Classic Chrome"],
  ["classic negative", "Classic Negative"],
  ["nostalgic", "Nostalgic Neg"],
  ["eterna bleach", "Eterna Bleach Bypass"],
  ["eterna", "Eterna"],
  ["pro neg. std", "Pro Neg Std"],
  ["pro neg std", "Pro Neg Std"],
  ["pro neg. hi", "Pro Neg Hi"],
  ["pro neg hi", "Pro Neg Hi"],
  ["velvia", "Velvia"],
  ["reala ace", "Reala Ace"],
  ["astia", "Astia"],
  ["acros", "Acros"],
  ["monochrome", "Monochrome"],
  ["sepia", "Sepia"],
  ["provia", "Provia"],
  ["standard", "Provia"],
];

function mapFilmModeToSimulation(description: string | undefined): BaseFilmSimulation | "Unknown" {
  if (!description) return "Unknown";
  const normalized = description.toLowerCase();
  const match = FILM_MODE_KEYWORDS.find(([keyword]) => normalized.includes(keyword));
  return match ? match[1] : "Unknown";
}

const WHITE_BALANCE_KEYWORDS: Array<[string, WhiteBalanceMode]> = [
  ["auto", "Auto"],
  ["daylight", "Daylight"],
  ["shade", "Shade"],
  ["fluorescent 1", "Fluorescent1"],
  ["fluorescent1", "Fluorescent1"],
  ["fluorescent 2", "Fluorescent2"],
  ["fluorescent2", "Fluorescent2"],
  ["fluorescent 3", "Fluorescent3"],
  ["fluorescent3", "Fluorescent3"],
  ["incandescent", "Incandescent"],
  ["underwater", "Underwater"],
  ["kelvin", "Kelvin"],
  ["color temperature", "Kelvin"],
];

function mapWhiteBalanceMode(description: string | undefined): WhiteBalanceMode | "Unknown" {
  if (!description) return "Unknown";
  const normalized = description.toLowerCase();
  const match = WHITE_BALANCE_KEYWORDS.find(([keyword]) => normalized.includes(keyword));
  return match ? match[1] : "Unknown";
}

/**
 * Fuji's WhiteBalanceFineTune tag is typically a (red, blue) pair in raw
 * units well outside the -9..+9 UI scale; /20 is a commonly cited scaling
 * factor in EXIF tooling for this tag, unverified here.
 */
const WB_FINE_TUNE_SCALE = 20;

function parseWhiteBalanceFineTune(value: number[] | number | undefined): WhiteBalanceShift {
  if (Array.isArray(value) && value.length >= 2) {
    return { red: value[0] / WB_FINE_TUNE_SCALE, blue: value[1] / WB_FINE_TUNE_SCALE };
  }
  return { red: 0, blue: 0 };
}

/**
 * Fuji's HighlightTone/ShadowTone/Sharpness tags are commonly reported in
 * doubled units (raw -4..+8 for a UI range of -2..+4); /2 is a starting
 * assumption, unverified here.
 */
const TONE_SCALE = 2;

function parseToneValue(value: number | undefined): number {
  if (typeof value !== "number") return 0;
  return value / TONE_SCALE;
}

/**
 * Extracts the camera's baked-in film simulation settings from a Fuji
 * JPEG's EXIF MakerNotes. Returns null if the file has no Fuji MakerNotes
 * (non-Fuji camera, or EXIF stripped by prior editing) — callers fall back
 * to a neutral baseline (see computeRecipeAdjustment in lib/recipes/neutralize.ts).
 */
export async function extractDetectedSettings(file: File): Promise<DetectedSettings | null> {
  let tags: ExpandedExifTags;
  try {
    tags = (await ExifReader.load(file, { expanded: true })) as ExpandedExifTags;
  } catch {
    return null;
  }

  const maker = tags.Fujifilm;
  if (!maker) return null;

  return {
    cameraModel: tags.exif?.Model?.description ?? null,
    baseFilmSimulation: mapFilmModeToSimulation(maker.FilmMode?.description ?? maker.Saturation?.description),
    whiteBalance: {
      mode: mapWhiteBalanceMode(maker.WhiteBalance?.description),
      shift: parseWhiteBalanceFineTune(maker.WhiteBalanceFineTune?.value),
    },
    highlightTone: parseToneValue(maker.HighlightTone?.value ?? maker.Highlight?.value),
    shadowTone: parseToneValue(maker.ShadowTone?.value ?? maker.Shadow?.value),
    color: parseToneValue(maker.Color?.value ?? maker.Saturation?.value),
    sharpness: parseToneValue(maker.Sharpness?.value),
  };
}
