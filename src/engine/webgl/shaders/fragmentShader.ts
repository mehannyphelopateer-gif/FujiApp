/**
 * Recipe render pipeline, in order:
 *   1. Sharpness (forward-only convolution on the raw source texture — see
 *      src/lib/recipes/neutralize.ts for why this isn't neutralized like the
 *      other axes; run first because it needs neighbor samples of the raw
 *      image, matching how a camera sharpens before applying film science).
 *   2. Base Film Simulation swap via the Hald CLUT (u_lutTexture).
 *   3. White balance shift.
 *   4. Highlight/shadow tone curve.
 *   5. Saturation (Color).
 *   6. Color Chrome Effect (saturated warm-hue luminance compression).
 *   7. Color Chrome FX Blue (same idea, blue-weighted; X-Trans IV+ only).
 *   8. Grain overlay (static per-recipe noise, not animated per-frame; size
 *      controls the noise frequency, i.e. blob size).
 *
 * All the numeric curves here (tone curve, saturation, color chrome, grain)
 * are parametric approximations of Fuji's in-camera processing, not exact
 * fits to their published color science — tunable later once compared side
 * by side against real X Weekly reference JPEGs.
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
uniform float u_lutSize;      // levels per channel, 64.0 for a level-8 Hald CLUT
uniform vec2 u_texelSize;     // 1.0 / canvas size, for the sharpness convolution
uniform float u_sharpness;    // forward-only target, roughly -4..4
uniform vec2 u_wbShift;       // (red, blue) neutralized delta, roughly -18..18
uniform float u_highlightTone;  // neutralized delta
uniform float u_shadowTone;     // neutralized delta
uniform float u_saturation;     // neutralized delta (Color), roughly -8..8
uniform float u_colorChromeStrength;    // 0.0 (Off) / 0.5 (Weak) / 1.0 (Strong)
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
vec3 applyWhiteBalance(vec3 color, vec2 shift) {
  // shift is on Fuji's UI scale (roughly -9..+9 per axis, larger after
  // neutralization deltas). Scaled down to a subtle per-channel gain.
  float redGain = 1.0 + shift.x * 0.015;
  float blueGain = 1.0 + shift.y * 0.015;
  return vec3(color.r * redGain, color.g, color.b * blueGain);
}

// ---- Highlight / Shadow tone curve ----
vec3 applyToneCurve(vec3 color, float highlight, float shadow) {
  float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
  float hAmt = clamp(highlight / 6.0, -1.0, 1.0);
  float sAmt = clamp(shadow / 6.0, -1.0, 1.0);

  float highlightWeight = smoothstep(0.5, 1.0, luma);
  float shadowWeight = 1.0 - smoothstep(0.0, 0.5, luma);

  vec3 result = color * (1.0 - highlightWeight * hAmt * 0.5);
  result += shadowWeight * sAmt * 0.15;
  return clamp(result, 0.0, 1.0);
}

// ---- Saturation (Color) ----
vec3 applySaturation(vec3 color, float delta) {
  if (abs(delta) < 0.001) return color;
  float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
  float factor = 1.0 + clamp(delta / 8.0, -1.0, 1.0);
  return clamp(mix(vec3(luma), color, factor), 0.0, 1.0);
}

// ---- Color Chrome Effect / FX Blue: compress luminance, deepen saturation
// on highly-saturated pixels of a target hue (warm for the base effect,
// blue for FX Blue — X-Trans IV+ cameras run both independently). ----
vec3 applyColorChrome(vec3 color, float strength, float hueWeight) {
  if (strength <= 0.0) return color;

  float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
  float maxC = max(color.r, max(color.g, color.b));
  float minC = min(color.r, min(color.g, color.b));
  float sat = maxC - minC;

  float amount = sat * hueWeight * strength;

  float lumaCompressed = luma - amount * 0.15;
  vec3 desaturated = vec3(luma);
  vec3 boosted = mix(desaturated, color, 1.0 + amount * 0.35);
  float lumaBoosted = max(dot(boosted, vec3(0.2126, 0.7152, 0.0722)), 0.0001);

  return clamp(boosted * (lumaCompressed / lumaBoosted), 0.0, 1.0);
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
  vec3 simColor = apply3DLut(sharpened, u_lutTexture, u_lutSize);
  vec3 wbColor = applyWhiteBalance(simColor, u_wbShift);
  vec3 toneColor = applyToneCurve(wbColor, u_highlightTone, u_shadowTone);
  vec3 satColor = applySaturation(toneColor, u_saturation);

  // Warm-hue weight favors high R/G relative to B; blue weight favors the reverse.
  float warmWeight = clamp(satColor.r + satColor.g * 0.5 - satColor.b, 0.0, 1.0);
  float blueWeight = clamp(satColor.b - max(satColor.r, satColor.g) * 0.5, 0.0, 1.0);
  vec3 chromeColor = applyColorChrome(satColor, u_colorChromeStrength, warmWeight);
  vec3 chromeBlueColor = applyColorChrome(chromeColor, u_colorChromeFxBlueStrength, blueWeight);

  vec3 finalColor = applyGrain(chromeBlueColor, gl_FragCoord.xy, u_grainStrength, u_grainSeed, u_grainSize);

  gl_FragColor = vec4(finalColor, alpha);
}
`;
