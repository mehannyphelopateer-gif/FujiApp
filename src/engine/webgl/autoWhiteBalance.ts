/**
 * Approximates a real camera's Auto White Balance correction, which this
 * app's RAW-conversion pipeline never performs — patchRawProfile.ts always
 * uses as-shot WB, and Phase 3's calibrated WB gain (parametricCalibration.ts)
 * only ever modeled the small ± SHIFT dial, not the much larger illuminant
 * correction Auto WB applies before that. A real camera JPEG already has
 * this baked in; only this app's own neutral RAW decode is missing it,
 * which is invisible on mildly-lit scenes but produces a dramatically
 * over-warm/uncorrected render on a scene shot under a strong single-
 * color-temperature light (confirmed by directly comparing pixel values
 * against a real X RAW Studio conversion of the same RAF — see
 * ~/.claude/plans/indexed-inventing-wren.md's Phase 3 "Round 7").
 *
 * Uses a gray-world estimate (assume the scene's average color, excluding
 * near-black/near-white outliers, should be neutral) damped to 50%
 * strength. Full-strength gray-world overshoots in some regions and
 * undershoots in others relative to the real camera's own correction —
 * validated by direct pixel-error comparison against the real reference:
 * identity error 19.1, full-strength 16.8, 50%-damped 11.75 (the measured
 * sweet spot). Like any gray-world AWB, this can misfire on a scene with a
 * legitimately dominant single hue (a sunset, a forest) — a known,
 * inherent limitation of the technique, not specific to this
 * implementation.
 */
const AWB_DAMPING = 0.5;
const NEAR_BLACK = 10;
const NEAR_WHITE = 245;
const SAMPLE_SIZE = 64;

export interface AwbGain {
  red: number;
  blue: number;
}

export const IDENTITY_AWB_GAIN: AwbGain = { red: 1, blue: 1 };

/** Estimates the gray-world AWB gain from an already-loaded image's actual pixel content. */
export function estimateAwbGain(image: HTMLImageElement): AwbGain {
  const canvas = document.createElement("canvas");
  canvas.width = SAMPLE_SIZE;
  canvas.height = SAMPLE_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) return IDENTITY_AWB_GAIN;

  ctx.drawImage(image, 0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
  let data: Uint8ClampedArray;
  try {
    data = ctx.getImageData(0, 0, SAMPLE_SIZE, SAMPLE_SIZE).data;
  } catch {
    return IDENTITY_AWB_GAIN; // tainted canvas (cross-origin source) — no-op fallback
  }

  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let n = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    if (r < NEAR_BLACK && g < NEAR_BLACK && b < NEAR_BLACK) continue;
    if (r > NEAR_WHITE || g > NEAR_WHITE || b > NEAR_WHITE) continue;
    sumR += r;
    sumG += g;
    sumB += b;
    n++;
  }
  if (n === 0) return IDENTITY_AWB_GAIN;

  const avgR = sumR / n;
  const avgG = sumG / n;
  const avgB = sumB / n;
  const rawRedGain = avgG / avgR;
  const rawBlueGain = avgG / avgB;

  return {
    red: 1 + (rawRedGain - 1) * AWB_DAMPING,
    blue: 1 + (rawBlueGain - 1) * AWB_DAMPING,
  };
}
