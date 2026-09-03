import {
  WB_RED_GAIN_CURVE,
  WB_BLUE_GAIN_CURVE,
  HIGHLIGHT_TONE_CURVE,
  SHADOW_TONE_CURVE,
  SATURATION_FACTOR_CURVE,
} from "@/engine/webgl/generated/wbToneSaturationCurves";

/**
 * Piecewise-linear interpolation over a sorted set of {x, y} control
 * points, clamping to the nearest endpoint outside the tested range. Used
 * for every Phase 3 calibrated curve below — see
 * ~/.claude/plans/indexed-inventing-wren.md's Phase 3 section for how the
 * control points themselves are measured against real camera output
 * (scripts/derive-parametric-curves.mjs). This mirrors the exact
 * "JS-evaluated-once-per-draw-call" pattern useWebGLRenderer.ts's
 * lerpEffectStrength already uses for grain — a plain multiply/curve
 * lookup in TS, not GLSL.
 */
function interpolate(points: { x: number; y: number }[], x: number): number {
  if (points.length === 0) return x; // shouldn't happen — every curve always has at least the zero point
  if (x <= points[0].x) return points[0].y;
  const last = points[points.length - 1];
  if (x >= last.x) return last.y;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    if (x >= a.x && x <= b.x) {
      const t = b.x === a.x ? 0 : (x - a.x) / (b.x - a.x);
      return a.y + (b.y - a.y) * t;
    }
  }
  return last.y;
}

const wbRedPoints = WB_RED_GAIN_CURVE.map((p) => ({ x: p.shift, y: p.gain }));
const wbBluePoints = WB_BLUE_GAIN_CURVE.map((p) => ({ x: p.shift, y: p.gain }));
const highlightPoints = HIGHLIGHT_TONE_CURVE.map((p) => ({ x: p.value, y: p.amount }));
const shadowPoints = SHADOW_TONE_CURVE.map((p) => ({ x: p.value, y: p.amount }));
const saturationPoints = SATURATION_FACTOR_CURVE.map((p) => ({ x: p.value, y: p.factor }));

/** Calibrated per-channel multiplicative gain for a requested WB shift — replaces applyWhiteBalance's old `1 + shift*0.015` guess. */
export function getWbGain(shift: { red: number; blue: number }): { red: number; blue: number } {
  return {
    red: interpolate(wbRedPoints, shift.red),
    blue: interpolate(wbBluePoints, shift.blue),
  };
}

/** Calibrated highlight-tone amount (fed directly into applyToneCurve's existing luma-zone-weighted shape) for a requested highlightTone delta. */
export function getHighlightAmount(highlightTone: number): number {
  return interpolate(highlightPoints, highlightTone);
}

/** Calibrated shadow-tone amount (fed directly into applyToneCurve's existing luma-zone-weighted shape) for a requested shadowTone delta. */
export function getShadowAmount(shadowTone: number): number {
  return interpolate(shadowPoints, shadowTone);
}

/** Calibrated saturation blend factor (fed directly into applySaturation's existing luma-preserving mix) for a requested color/saturation delta. */
export function getSaturationFactor(color: number): number {
  return interpolate(saturationPoints, color);
}
