import type { BaseFilmSimulation } from "@/types/recipe";

type GLContext = WebGL2RenderingContext | WebGLRenderingContext;

/** Levels per channel for the standard Hald CLUT assets in /public/luts (level-8, 512x512). */
export const LUT_SIZE = 64;

export const LUT_MANIFEST: Partial<Record<BaseFilmSimulation, string>> = {
  "Classic Chrome": "/luts/classic-chrome.png",
  "Classic Negative": "/luts/classic-negative.png",
  "Pro Neg Std": "/luts/pro-neg-std.png",
  Velvia: "/luts/velvia.png",
  Acros: "/luts/acros.png",
  "Nostalgic Neg": "/luts/nostalgic-neg.png",
  Provia: "/luts/provia.png",
  Astia: "/luts/astia.png",
  "Pro Neg Hi": "/luts/pro-neg-hi.png",
  Eterna: "/luts/eterna.png",
  "Eterna Bleach Bypass": "/luts/eterna-bleach-bypass.png",
  Monochrome: "/luts/monochrome.png",
  Sepia: "/luts/sepia.png",
  "Reala Ace": "/luts/reala-ace.png",
};

export const IDENTITY_LUT_URL = "/luts/identity.png";

// Fitted via scripts/invert-luts.mjs — undoes a film simulation already
// baked into a source photo before the shader applies a different target
// simulation's own LUT (see neutralize.ts's sourceFilmSimulationToUndo).
// Acros/Monochrome/Sepia have no entry: undoing a monochrome/toned render
// is physically impossible (no color left to recover), not a missing asset.
export const INVERSE_LUT_MANIFEST: Partial<Record<BaseFilmSimulation, string>> = {
  "Classic Chrome": "/luts/inverse/classic-chrome.png",
  "Classic Negative": "/luts/inverse/classic-negative.png",
  "Pro Neg Std": "/luts/inverse/pro-neg-std.png",
  Velvia: "/luts/inverse/velvia.png",
  "Nostalgic Neg": "/luts/inverse/nostalgic-neg.png",
  Provia: "/luts/inverse/provia.png",
  Astia: "/luts/inverse/astia.png",
  "Pro Neg Hi": "/luts/inverse/pro-neg-hi.png",
  Eterna: "/luts/inverse/eterna.png",
  "Eterna Bleach Bypass": "/luts/inverse/eterna-bleach-bypass.png",
  "Reala Ace": "/luts/inverse/reala-ace.png",
};

const imageCache = new Map<string, Promise<HTMLImageElement>>();

function loadLutImage(url: string): Promise<HTMLImageElement> {
  let cached = imageCache.get(url);
  if (!cached) {
    cached = new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(`Failed to load LUT image: ${url}`));
      image.src = url;
    });
    imageCache.set(url, cached);
  }
  return cached;
}

export function resolveLutUrl(baseFilmSimulation: BaseFilmSimulation): string {
  return LUT_MANIFEST[baseFilmSimulation] ?? IDENTITY_LUT_URL;
}

/** `undefined` (nothing to undo) resolves to the identity LUT, same as an unmapped simulation. */
export function resolveInverseLutUrl(sourceFilmSimulationToUndo?: BaseFilmSimulation): string {
  if (!sourceFilmSimulationToUndo) return IDENTITY_LUT_URL;
  return INVERSE_LUT_MANIFEST[sourceFilmSimulationToUndo] ?? IDENTITY_LUT_URL;
}

async function createTextureFromUrl(gl: GLContext, url: string): Promise<WebGLTexture> {
  const image = await loadLutImage(url);
  const texture = gl.createTexture();
  if (!texture) throw new Error("Failed to create LUT texture");
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  // CLAMP_TO_EDGE is required: with REPEAT, hardware bilinear filtering would
  // blend the last row of one blue slice with the first row of the next.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return texture;
}

export function createLutTexture(gl: GLContext, baseFilmSimulation: BaseFilmSimulation): Promise<WebGLTexture> {
  return createTextureFromUrl(gl, resolveLutUrl(baseFilmSimulation));
}

export function createInverseLutTexture(gl: GLContext, sourceFilmSimulationToUndo?: BaseFilmSimulation): Promise<WebGLTexture> {
  return createTextureFromUrl(gl, resolveInverseLutUrl(sourceFilmSimulationToUndo));
}
