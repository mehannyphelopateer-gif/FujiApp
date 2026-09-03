// DEFAULT/FALLBACK control points — no real Phase 3 calibration shoot has
// been run yet, so these exactly reproduce today's hand-tuned shader
// formulas (fragmentShader.ts's old applyWhiteBalance/applyToneCurve/
// applySaturation constants: WB gain = 1 + shift*0.015, tone amount =
// clamp(value/6, -1, 1), saturation factor = 1 + clamp(value/8, -1, 1)) so
// switching to the calibrated-curve code path is a zero-behavior-change
// no-op until real data replaces this file.
//
// Once a Phase 3 calibration shoot exists, run
// `node scripts/derive-parametric-curves.mjs` to OVERWRITE this file with
// real measured control points — see
// ~/.claude/plans/indexed-inventing-wren.md's Phase 3 section.

export interface WbGainPoint {
  shift: number;
  gain: number;
}

export interface ToneAmountPoint {
  value: number;
  amount: number;
}

export interface SaturationFactorPoint {
  value: number;
  factor: number;
}

export const WB_RED_GAIN_CURVE: WbGainPoint[] = [
  { shift: -9, gain: 0.865 },
  { shift: -4, gain: 0.94 },
  { shift: 0, gain: 1 },
  { shift: 4, gain: 1.06 },
  { shift: 9, gain: 1.135 },
];

export const WB_BLUE_GAIN_CURVE: WbGainPoint[] = [
  { shift: -9, gain: 0.865 },
  { shift: -4, gain: 0.94 },
  { shift: 0, gain: 1 },
  { shift: 4, gain: 1.06 },
  { shift: 9, gain: 1.135 },
];

export const HIGHLIGHT_TONE_CURVE: ToneAmountPoint[] = [
  { value: -2, amount: -1 / 3 },
  { value: 0, amount: 0 },
  { value: 2, amount: 1 / 3 },
  { value: 4, amount: 2 / 3 },
];

export const SHADOW_TONE_CURVE: ToneAmountPoint[] = [
  { value: -2, amount: -1 / 3 },
  { value: 0, amount: 0 },
  { value: 2, amount: 1 / 3 },
  { value: 4, amount: 2 / 3 },
];

export const SATURATION_FACTOR_CURVE: SaturationFactorPoint[] = [
  { value: -4, factor: 0.5 },
  { value: -2, factor: 0.75 },
  { value: 0, factor: 1 },
  { value: 2, factor: 1.25 },
  { value: 4, factor: 1.5 },
];
