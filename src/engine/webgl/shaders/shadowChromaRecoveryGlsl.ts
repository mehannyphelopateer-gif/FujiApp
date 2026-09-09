import { SHADOW_BLUE_LIFT_CURVE } from "@/engine/webgl/generated/shadowChromaRecovery";

/**
 * Emits a fixed GLSL function that piecewise-linearly interpolates the
 * calibrated shadow blue-recovery curve — baked in as compile-time literal
 * constants (the curve is a fixed, app-wide calibration constant, not a
 * per-recipe value, so there's no need to plumb it through as uniforms the
 * way every other Phase 3 axis is). See autoWhiteBalance.ts's doc comment
 * and ~/.claude/plans/indexed-inventing-wren.md's Phase 3 "Round 16" for
 * why this correction exists and why it's additive, not multiplicative.
 */
export function buildShadowBlueLiftGlsl(): string {
  const points = SHADOW_BLUE_LIFT_CURVE;
  const branches = points
    .slice(0, -1)
    .map((point, i) => {
      const next = points[i + 1];
      return `  if (luma <= ${next.luma.toFixed(6)}) {
    float t = (luma - ${point.luma.toFixed(6)}) / ${(next.luma - point.luma).toFixed(6)};
    return mix(${point.blueLift.toFixed(6)}, ${next.blueLift.toFixed(6)}, t);
  }`;
    })
    .join("\n");
  const last = points[points.length - 1];

  return `float shadowBlueLift(float luma) {
  if (luma <= ${points[0].luma.toFixed(6)}) return ${points[0].blueLift.toFixed(6)};
${branches}
  return ${last.blueLift.toFixed(6)};
}`;
}
