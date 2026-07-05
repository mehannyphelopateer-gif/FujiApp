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

export async function createLutTexture(
  gl: GLContext,
  baseFilmSimulation: BaseFilmSimulation,
): Promise<WebGLTexture> {
  const image = await loadLutImage(resolveLutUrl(baseFilmSimulation));
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
