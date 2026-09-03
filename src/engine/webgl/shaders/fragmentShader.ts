/**
 * Recipe render pipeline, in order:
 *   1. Sharpness (forward-only convolution on the raw source texture — see
 *      src/lib/recipes/neutralize.ts for why this isn't neutralized like the
 *      other axes; run first because it needs neighbor samples of the raw
 *      image, matching how a camera sharpens before applying film science).
 *   2. Undo a source film simulation already baked in, if any
 *      (u_sourceInverseLutTexture — identity when there's nothing to undo),
 *      then swap in the target Base Film Simulation (u_lutTexture). Without
 *      the undo pass, picking a new recipe on a photo that already has one
 *      baked in would stack two film simulations instead of swapping one
 *      for the other — see neutralize.ts's sourceFilmSimulationToUndo and
 *      scripts/invert-luts.mjs for how the inverse LUTs are derived.
 *   3. White balance shift (calibrated gain — see parametricCalibration.ts).
 *   4. Highlight/shadow tone curve (calibrated amount, same shape).
 *   5. Saturation (Color) (calibrated factor, same shape).
 *   6. Color Chrome Effect (calibrated Hald CLUT, warm-hue weighted).
 *   7. Color Chrome FX Blue (calibrated Hald CLUT, blue-hue weighted;
 *      X-Trans IV+ only).
 *   8. Grain overlay (static per-recipe noise, not animated per-frame; size
 *      controls the noise frequency, i.e. blob size — strength/size still
 *      hand-picked constants pending scripts/derive-grain-stats.mjs).
 *
 * Steps 3-5 (white balance, tone curve, saturation) keep their original
 * parametric SHAPE (luma-zone weighting, luma-preserving mix) but now take
 * an already-calibrated coefficient computed in
 * src/engine/webgl/parametricCalibration.ts from real camera measurements
 * (scripts/derive-parametric-curves.mjs), rather than deriving that
 * coefficient from a hand-picked division constant in this file. Steps 6-7
 * were fully replaced with calibrated Hald CLUTs (scripts/derive-color-
 * chrome-luts.mjs) since Color Chrome Effect/FX Blue are discrete
 * Off/Weak/Strong camera states, not a continuous dial — see
 * neutralize.ts's neutralizedStrength() invariant this depends on. Grain
 * (step 8) is still a hand-tuned approximation pending
 * scripts/derive-grain-stats.mjs's noise-statistics calibration.
 *
 * noiseReduction/isoRange/exposureCompensation from Recipe are deliberately
 * NOT uniforms here — they're capture-time camera settings, not something a
 * post-process shader can apply to an already-rendered JPEG.
 */
export const fragmentShaderSource = `
precision highp float;

varying vec2 v_texCoord;

uniform sampler2D u_image;
uniform sampler2D u_lutTexture;
uniform sampler2D u_sourceInverseLutTexture; // undoes a baked-in source film sim; identity when there's nothing to undo
uniform sampler2D u_colorChromeWeakLutTexture;
uniform sampler2D u_colorChromeStrongLutTexture;
uniform sampler2D u_fxBlueWeakLutTexture;
uniform sampler2D u_fxBlueStrongLutTexture;
uniform float u_lutSize;      // levels per channel, 64.0 for a level-8 Hald CLUT
uniform vec2 u_texelSize;     // 1.0 / canvas size, for the sharpness convolution
uniform float u_sharpness;    // forward-only target, roughly -4..4
uniform vec2 u_wbGain;        // calibrated (red, blue) multiplicative gain — see parametricCalibration.ts's getWbGain
uniform float u_highlightAmount; // calibrated amount, already in the same -1..1 scale applyToneCurve expects
uniform float u_shadowAmount;    // calibrated amount, already in the same -1..1 scale applyToneCurve expects
uniform float u_saturationFactor; // calibrated blend factor — see parametricCalibration.ts's getSaturationFactor
uniform float u_colorChromeStrength;    // 0.0 (Off) / 0.5 (Weak) / 1.0 (Strong) — see neutralize.ts's neutralizedStrength
uniform float u_colorChromeFxBlueStrength; // 0.0 (Off) / 0.5 (Weak) / 1.0 (Strong)
uniform float u_grainStrength;       // 0.0 (Off) / 0.035 (Weak) / 0.08 (Strong)
uniform float u_grainSize;           // noise-coordinate scale: bigger = finer grain
uniform float u_grainSeed;

// ---- Sharpness (forward-only unsharp mask / soft blur) ----
vec3 applySharpness(sampler2D tex, vec2 uv, vec2 texelSize, float amount) {
  vec3 center = texture2D(tex, uv).rgb;
  if (abs(amount) < 0.001) return center;

  vec3 neighborSum =
    texture2D(tex, uv + vec2(-texelSize.x, 0.0)).rgb +
    texture2D(tex, uv + vec2( texelSize.x, 0.0)).rgb +
    texture2D(tex, uv + vec2(0.0, -texelSize.y)).rgb +
    texture2D(tex, uv + vec2(0.0,  texelSize.y)).rgb;

  if (amount > 0.0) {
    vec3 blurred = neighborSum * 0.25;
    return clamp(center + (center - blurred) * amount * 0.5, 0.0, 1.0);
  }

  vec3 blurred = (neighborSum + center) * 0.2;
  return mix(center, blurred, clamp(-amount * 0.5, 0.0, 1.0));
}

// ---- Base Film Simulation: standard Hald CLUT sample (512x512, 64 levels/channel) ----
// UV for one (r, g) level pair within a single blue-slice band.
// n = sqrt(levels) = rows per slice (8.0 for levels = 64.0).
//
// g is snapped to its nearest integer level (floor(g + 0.5)) before being
// split into gHi/gLo. Without this, a fractional g bakes directly into the
// v coordinate; since adjacent texel *rows* in this packing represent g
// values a full octave apart (±n, not ±1), hardware bilinear filtering
// would blend across that boundary and produce a jarring striped/moiré
// pattern on any smooth green gradient instead of a clean image — caught by
// actually rendering a real photo, not just by reading the math. Snapping
// g means the v coordinate always lands exactly on a texel-row center, so
// there's nothing for vertical bilinear filtering to incorrectly blend.
vec2 haldUV(vec2 rg, float bSlice, float levels, float n, float side) {
  float gLevel = floor(rg.y + 0.5);
  float gHi = floor(gLevel / n);
  float gLo = mod(gLevel, n);
  float x = rg.x + gLo * levels;
  float y = gHi + bSlice * n;
  return vec2((x + 0.5) / side, (y + 0.5) / side);
}

vec3 apply3DLut(vec3 color, sampler2D lutTex, float levels) {
  float maxLevel = levels - 1.0;
  vec3 lutCoord = clamp(color, 0.0, 1.0) * maxLevel;

  float n = sqrt(levels);
  float side = levels * n;

  float cell = floor(lutCoord.b);
  float cell2 = min(cell + 1.0, maxLevel);
  float frac = lutCoord.b - cell;

  vec2 uv1 = haldUV(lutCoord.rg, cell, levels, n, side);
  vec2 uv2 = haldUV(lutCoord.rg, cell2, levels, n, side);

  vec3 sampledColor1 = texture2D(lutTex, uv1).rgb;
  vec3 sampledColor2 = texture2D(lutTex, uv2).rgb;
  return mix(sampledColor1, sampledColor2, frac); // linear across the blue axis
}

// ---- White Balance shift ----
// TRIED AND REVERTED: applying this gain in linear light (converting via
// the sRGB transfer function before/after) instead of directly to these
// gamma-encoded values, on the theory that WB is physically a linear-light
// correction on a real camera. Real-hardware evidence contradicted the
// theory: worked through the actual math (see git history/PR for the
// numbers) and linear-space application makes gamma-space shadows MORE
// affected and highlights LESS affected than a direct multiply — backwards
// from what was needed for a dark, shadow-dominated test scene, and the
// user's own side-by-side comparison confirmed it made that case visibly
// MORE wrong, not less. Whatever this camera's real internal WB/tone-curve
// order is, a naive linear-light model doesn't match its actual gamma-
// space output — reverted to the simpler, empirically-correct direct
// multiply. scripts/derive-parametric-curves.mjs's measureChannelGain
// matches this (measures the ratio directly in gamma-encoded pixel space).
vec3 applyWhiteBalance(vec3 color, vec2 gain) {
  return vec3(color.r * gain.x, color.g, color.b * gain.y);
}

// ---- Highlight / Shadow tone curve ----
vec3 applyToneCurve(vec3 color, float hAmt, float sAmt) {
  float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));

  float highlightWeight = smoothstep(0.5, 1.0, luma);
  float shadowWeight = 1.0 - smoothstep(0.0, 0.5, luma);

  vec3 result = color * (1.0 - highlightWeight * hAmt * 0.5);
  result += shadowWeight * sAmt * 0.15;
  return clamp(result, 0.0, 1.0);
}

// ---- Saturation (Color) ----
vec3 applySaturation(vec3 color, float factor) {
  if (abs(factor - 1.0) < 0.001) return color;
  float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
  return clamp(mix(vec3(luma), color, factor), 0.0, 1.0);
}

// ---- Color Chrome Effect / FX Blue: calibrated Hald CLUTs at the camera's
// real Weak/Strong states. fraction (0.0/0.5/1.0 — see neutralize.ts's
// neutralizedStrength(), which is mathematically ALWAYS exactly one of
// these three values for any Off/Weak/Strong pair, never in between)
// selects which real state to sample instead of true-blending between them
// — cheaper than a 3-texture blend, and correct as long as that invariant
// holds. If neutralizedStrength() ever gains a 4th intermediate value, this
// needs revisiting into a real blend instead of a cutover. ----
vec3 applyChromeLut(vec3 color, float fraction, sampler2D weakLut, sampler2D strongLut, float levels) {
  if (fraction <= 0.0) return color;
  return fraction <= 0.5 ? apply3DLut(color, weakLut, levels) : apply3DLut(color, strongLut, levels);
}

// ---- Grain: static per-recipe procedural noise (not animated per-frame) ----
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123);
}

vec3 applyGrain(vec3 color, vec2 fragCoord, float strength, float seed, float sizeScale) {
  if (strength <= 0.0) return color;
  float n = hash(fragCoord * sizeScale + seed) - 0.5;
  return clamp(color + vec3(n * strength), 0.0, 1.0);
}

void main() {
  float alpha = texture2D(u_image, v_texCoord).a;

  vec3 sharpened = applySharpness(u_image, v_texCoord, u_texelSize, u_sharpness);
  vec3 undoneColor = apply3DLut(sharpened, u_sourceInverseLutTexture, u_lutSize);
  vec3 simColor = apply3DLut(undoneColor, u_lutTexture, u_lutSize);
  vec3 wbColor = applyWhiteBalance(simColor, u_wbGain);
  vec3 toneColor = applyToneCurve(wbColor, u_highlightAmount, u_shadowAmount);
  vec3 satColor = applySaturation(toneColor, u_saturationFactor);

  vec3 chromeColor = applyChromeLut(satColor, u_colorChromeStrength, u_colorChromeWeakLutTexture, u_colorChromeStrongLutTexture, u_lutSize);
  vec3 chromeBlueColor = applyChromeLut(chromeColor, u_colorChromeFxBlueStrength, u_fxBlueWeakLutTexture, u_fxBlueStrongLutTexture, u_lutSize);

  vec3 finalColor = applyGrain(chromeBlueColor, gl_FragCoord.xy, u_grainStrength, u_grainSeed, u_grainSize);

  gl_FragColor = vec4(finalColor, alpha);
}
`;
