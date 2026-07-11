import type { BaseFilmSimulation, DynamicRange, EffectStrength, GrainSize, Recipe } from "@/types/recipe";
import type { CameraSlotRaw } from "@/types/camera";

/**
 * Maps camera preset property values (D18E-D1A5) to Recipe-shaped fields.
 * Ported from filmkit's src/profile/preset-translate.ts and
 * src/profile/enums.ts — see ios/App/App/CameraLink/FujiPTPConstants.swift
 * for the Swift-side mirror of these same tables. Preset property encoding
 * (1-indexed effects, flat grain enum, raw DR%) differs from the RAW-
 * conversion d185 profile format — don't reuse these maps for that.
 */

const FILM_SIM_MAP: Record<number, BaseFilmSimulation> = {
  0x01: "Provia",
  0x02: "Velvia",
  0x03: "Astia",
  0x04: "Pro Neg Hi",
  0x05: "Pro Neg Std",
  0x06: "Monochrome",
  0x07: "Monochrome", // + Yellow filter — FujiApp's Recipe type has no distinct variant
  0x08: "Monochrome", // + Red filter
  0x09: "Monochrome", // + Green filter
  0x0a: "Sepia",
  0x0b: "Classic Chrome",
  0x0c: "Acros",
  0x0d: "Acros", // + Yellow filter
  0x0e: "Acros", // + Red filter
  0x0f: "Acros", // + Green filter
  0x10: "Eterna",
  0x11: "Classic Negative",
  0x12: "Eterna Bleach Bypass",
  0x13: "Nostalgic Neg",
  0x14: "Reala Ace",
};

/** Fuji WB mode values -> FujiApp's WhiteBalanceMode. AsShot/AmbiencePriority
 *  have no direct equivalent in the Recipe model and fall back to "Auto". */
const WB_MODE_MAP: Record<number, Recipe["whiteBalance"]["mode"]> = {
  0x0000: "Auto",
  0x0002: "Auto",
  0x0004: "Daylight",
  0x0006: "Incandescent",
  0x0008: "Underwater",
  0x8001: "Fluorescent1",
  0x8002: "Fluorescent2",
  0x8003: "Fluorescent3",
  0x8006: "Shade",
  0x8007: "Kelvin",
  0x8021: "Auto",
};

const DR_MAP: Record<number, DynamicRange> = { 100: "DR100", 200: "DR200", 400: "DR400" };

/** Preset effect encoding is 1-indexed (1=Off, 2=Weak, 3=Strong) — not 0/1/2. */
const EFFECT_MAP: Record<number, EffectStrength> = { 1: "Off", 2: "Weak", 3: "Strong" };

/** Preset grain is a flat 1-5 enum (strength x size combined), not separate fields. */
const GRAIN_MAP: Record<number, { effect: EffectStrength; size?: GrainSize }> = {
  1: { effect: "Off" },
  2: { effect: "Weak", size: "Small" },
  3: { effect: "Strong", size: "Small" },
  4: { effect: "Weak", size: "Large" },
  5: { effect: "Strong", size: "Large" },
};

export interface DecodedCameraSlot {
  name: string;
  baseFilmSimulation: BaseFilmSimulation;
  dynamicRange: DynamicRange;
  whiteBalance: Recipe["whiteBalance"];
  highlightTone: number;
  shadowTone: number;
  color: number;
  sharpness: number;
  colorChromeEffect: EffectStrength;
  colorChromeFxBlue: EffectStrength;
  grainEffect: EffectStrength;
  grainSize?: GrainSize;
  clarity: number;
}

function num(raw: CameraSlotRaw, propId: string): number | undefined {
  const v = raw.properties[propId];
  return typeof v === "number" ? v : undefined;
}

/** Decode a x10-encoded tone value. 0x8000/-32768 is a sentinel for "not set". */
function decodeTone(raw: number | undefined): number {
  if (raw === undefined || raw === 0x8000 || raw === -32768) return 0;
  return raw / 10;
}

/** Decode one camera custom-setting slot (C1-C7) into Recipe-shaped fields. */
export function decodeCameraSlot(raw: CameraSlotRaw): DecodedCameraSlot {
  const filmSimRaw = num(raw, "D192") ?? 0x01;
  const wbModeRaw = (num(raw, "D199") ?? 0) & 0xffff;
  const wbMode = WB_MODE_MAP[wbModeRaw] ?? "Auto";
  const wbColorTemp = num(raw, "D19C") ?? 0;
  const grainRaw = num(raw, "D195") ?? 1;
  const grain = GRAIN_MAP[grainRaw] ?? { effect: "Off" as EffectStrength };

  return {
    name: raw.name,
    baseFilmSimulation: FILM_SIM_MAP[filmSimRaw] ?? "Provia",
    dynamicRange: DR_MAP[num(raw, "D190") ?? 100] ?? "DR100",
    whiteBalance: {
      mode: wbMode,
      ...(wbMode === "Kelvin" && wbColorTemp > 0 ? { kelvin: wbColorTemp } : {}),
      shift: {
        red: num(raw, "D19A") ?? 0,
        blue: num(raw, "D19B") ?? 0,
      },
    },
    highlightTone: decodeTone(num(raw, "D19D")),
    shadowTone: decodeTone(num(raw, "D19E")),
    color: decodeTone(num(raw, "D19F")),
    sharpness: decodeTone(num(raw, "D1A0")),
    colorChromeEffect: EFFECT_MAP[num(raw, "D196") ?? 1] ?? "Off",
    colorChromeFxBlue: EFFECT_MAP[num(raw, "D197") ?? 1] ?? "Off",
    grainEffect: grain.effect,
    grainSize: grain.size,
    clarity: decodeTone(num(raw, "D1A2")),
  };
}
