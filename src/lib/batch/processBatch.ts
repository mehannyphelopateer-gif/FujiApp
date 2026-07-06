import { createFullscreenQuad, createImageTexture, createProgram } from "@/engine/webgl/glUtils";
import { createLutTexture } from "@/engine/webgl/lut";
import { vertexShaderSource } from "@/engine/webgl/shaders/vertexShader";
import { fragmentShaderSource } from "@/engine/webgl/shaders/fragmentShader";
import { computeRecipeAdjustment } from "@/lib/recipes/neutralize";
import { extractDetectedSettings } from "@/lib/exif/parseFujiMakerNotes";
import type { Recipe } from "@/types/recipe";

const COLOR_CHROME_STRENGTH: Record<string, number> = { Off: 0, Weak: 0.5, Strong: 1 };
const GRAIN_STRENGTH: Record<string, number> = { Off: 0, Weak: 0.035, Strong: 0.08 };
const GRAIN_SIZE_SCALE: Record<string, number> = { Small: 0.9, Large: 0.35 };
const DEFAULT_GRAIN_SIZE_SCALE = 0.75;
const MAX_DIMENSION = 4096;

export interface BatchResult {
  name: string;
  blob: Blob;
}

function loadImage(file: File): Promise<{ image: HTMLImageElement; url: string }> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => resolve({ image, url });
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`Failed to decode ${file.name}`));
    };
    image.src = url;
  });
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Failed to encode output image"))),
      "image/jpeg",
      0.92,
    );
  });
}

/**
 * Applies one recipe to many photos in sequence, reusing a single hidden
 * canvas/WebGL context (recreating a context per photo would hit browser
 * context-count limits fast on a large batch). Each photo still gets its own
 * EXIF-based neutralization baseline, same as the single-photo flow — a
 * batch of photos from different bodies/settings isn't assumed uniform.
 */
export async function processBatch(
  files: File[],
  recipe: Recipe,
  onProgress?: (done: number, total: number) => void,
  maxDimension: number = MAX_DIMENSION,
): Promise<BatchResult[]> {
  const canvas = document.createElement("canvas");
  const gl = (canvas.getContext("webgl2") ?? canvas.getContext("webgl")) as
    | WebGL2RenderingContext
    | WebGLRenderingContext
    | null;
  if (!gl) throw new Error("WebGL is not supported in this browser.");

  const program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
  createFullscreenQuad(gl, program);
  gl.useProgram(program);

  const uniforms = {
    u_image: gl.getUniformLocation(program, "u_image"),
    u_lutTexture: gl.getUniformLocation(program, "u_lutTexture"),
    u_lutSize: gl.getUniformLocation(program, "u_lutSize"),
    u_texelSize: gl.getUniformLocation(program, "u_texelSize"),
    u_sharpness: gl.getUniformLocation(program, "u_sharpness"),
    u_wbShift: gl.getUniformLocation(program, "u_wbShift"),
    u_highlightTone: gl.getUniformLocation(program, "u_highlightTone"),
    u_shadowTone: gl.getUniformLocation(program, "u_shadowTone"),
    u_saturation: gl.getUniformLocation(program, "u_saturation"),
    u_colorChromeStrength: gl.getUniformLocation(program, "u_colorChromeStrength"),
    u_colorChromeFxBlueStrength: gl.getUniformLocation(program, "u_colorChromeFxBlueStrength"),
    u_grainStrength: gl.getUniformLocation(program, "u_grainStrength"),
    u_grainSize: gl.getUniformLocation(program, "u_grainSize"),
    u_grainSeed: gl.getUniformLocation(program, "u_grainSeed"),
  };

  // The recipe (and therefore its base film simulation) is the same for the
  // whole batch, so the LUT only needs loading once.
  const lutTexture = await createLutTexture(gl, recipe.baseFilmSimulation);

  const results: BatchResult[] = [];

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    let objectUrl: string | null = null;
    try {
      const detected = await extractDetectedSettings(file).catch(() => null);
      const adjustment = computeRecipeAdjustment(detected, recipe);

      const { image, url } = await loadImage(file);
      objectUrl = url;

      const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
      canvas.width = Math.round(image.naturalWidth * scale);
      canvas.height = Math.round(image.naturalHeight * scale);

      const imageTexture = createImageTexture(gl, image);

      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(program);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, imageTexture);
      gl.uniform1i(uniforms.u_image, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, lutTexture);
      gl.uniform1i(uniforms.u_lutTexture, 1);

      gl.uniform1f(uniforms.u_lutSize, 64.0);
      gl.uniform2f(uniforms.u_texelSize, 1 / canvas.width, 1 / canvas.height);
      gl.uniform1f(uniforms.u_sharpness, adjustment.sharpness);
      gl.uniform2f(uniforms.u_wbShift, adjustment.whiteBalanceShift.red, adjustment.whiteBalanceShift.blue);
      gl.uniform1f(uniforms.u_highlightTone, adjustment.highlightTone);
      gl.uniform1f(uniforms.u_shadowTone, adjustment.shadowTone);
      gl.uniform1f(uniforms.u_saturation, adjustment.color);
      gl.uniform1f(uniforms.u_colorChromeStrength, COLOR_CHROME_STRENGTH[adjustment.colorChromeEffect] ?? 0);
      gl.uniform1f(uniforms.u_colorChromeFxBlueStrength, COLOR_CHROME_STRENGTH[adjustment.colorChromeFxBlue] ?? 0);
      gl.uniform1f(uniforms.u_grainStrength, GRAIN_STRENGTH[adjustment.grainEffect] ?? 0);
      gl.uniform1f(
        uniforms.u_grainSize,
        adjustment.grainSize ? (GRAIN_SIZE_SCALE[adjustment.grainSize] ?? DEFAULT_GRAIN_SIZE_SCALE) : DEFAULT_GRAIN_SIZE_SCALE,
      );
      gl.uniform1f(uniforms.u_grainSeed, Math.random() * 1000);

      gl.drawArrays(gl.TRIANGLES, 0, 6);

      const blob = await canvasToBlob(canvas);
      const baseName = file.name.replace(/\.[^.]+$/, "");
      const recipeSlug = recipe.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
      results.push({ name: `${baseName}-${recipeSlug}.jpg`, blob });

      gl.deleteTexture(imageTexture);
    } finally {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      onProgress?.(i + 1, files.length);
    }
  }

  gl.deleteTexture(lutTexture);
  gl.deleteProgram(program);

  return results;
}
