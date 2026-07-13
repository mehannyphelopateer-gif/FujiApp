import { useEffect, useRef, useState, type RefObject } from "react";
import { createFullscreenQuad, createImageTexture, createProgram } from "@/engine/webgl/glUtils";
import { createLutTexture } from "@/engine/webgl/lut";
import { vertexShaderSource } from "@/engine/webgl/shaders/vertexShader";
import { fragmentShaderSource } from "@/engine/webgl/shaders/fragmentShader";
import type { RecipeAdjustment } from "@/lib/recipes/neutralize";

type GLContext = WebGL2RenderingContext | WebGLRenderingContext;

// Grain strength keyframes for fraction 0 / 0.5 / 1 (Off/Weak/Strong), matched
// against real recipe previews. recipeAdjustment.grainStrength is a
// neutralized fraction (see neutralize.ts) that can land between these when a
// photo already had some grain baked in, so it's interpolated, not looked up.
const GRAIN_STRENGTH_OFF = 0;
const GRAIN_STRENGTH_WEAK = 0.035;
const GRAIN_STRENGTH_STRONG = 0.08;
// Noise-coordinate scale: higher = more variation per pixel = finer grain.
const GRAIN_SIZE_SCALE: Record<string, number> = { Small: 0.9, Large: 0.35 };
const DEFAULT_GRAIN_SIZE_SCALE = 0.75;

/** Interpolates a 0..1 fraction across the Off/Weak/Strong keyframes above. */
function lerpEffectStrength(fraction: number, off: number, weak: number, strong: number): number {
  const clamped = Math.max(0, Math.min(1, fraction));
  return clamped <= 0.5 ? off + (weak - off) * (clamped / 0.5) : weak + (strong - weak) * ((clamped - 0.5) / 0.5);
}

// Most GPUs comfortably support 4096px textures; downscale anything larger
// so a full-res camera JPEG doesn't hit a hardware texture size limit.
const MAX_TEXTURE_DIMENSION = 4096;

interface UniformLocations {
  u_image: WebGLUniformLocation | null;
  u_lutTexture: WebGLUniformLocation | null;
  u_lutSize: WebGLUniformLocation | null;
  u_texelSize: WebGLUniformLocation | null;
  u_sharpness: WebGLUniformLocation | null;
  u_wbShift: WebGLUniformLocation | null;
  u_highlightTone: WebGLUniformLocation | null;
  u_shadowTone: WebGLUniformLocation | null;
  u_saturation: WebGLUniformLocation | null;
  u_colorChromeStrength: WebGLUniformLocation | null;
  u_colorChromeFxBlueStrength: WebGLUniformLocation | null;
  u_grainStrength: WebGLUniformLocation | null;
  u_grainSize: WebGLUniformLocation | null;
  u_grainSeed: WebGLUniformLocation | null;
}

interface RendererState {
  gl: GLContext;
  program: WebGLProgram;
  uniforms: UniformLocations;
  imageTexture: WebGLTexture | null;
  lutTexture: WebGLTexture | null;
}

interface UseWebGLRendererResult {
  isReady: boolean;
  error: string | null;
}

/**
 * Owns the GPU pipeline for one <canvas>: compiles the shader program once,
 * uploads the source image / LUT as textures when they change, and draws a
 * single frame whenever the recipe adjustment's uniforms change. There is no
 * continuous render loop — a still photo with static grain only needs to be
 * redrawn when something actually changes, which is what keeps recipe
 * switching feeling instantaneous.
 *
 * Also handles WebGL context loss/restore. This isn't just a defensive
 * nicety — mobile browsers (this is a PWA meant to be installed on a phone)
 * routinely reclaim GPU contexts when a tab is backgrounded, the screen
 * locks, or memory is tight, and iOS Safari in particular does this
 * aggressively. Without handling `webglcontextlost`/`webglcontextrestored`,
 * the canvas would show a blank/broken frame until the user force-reloads.
 */
export function useWebGLRenderer(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  imageUrl: string | null,
  recipeAdjustment: RecipeAdjustment | null,
  maxDimension: number = MAX_TEXTURE_DIMENSION,
): UseWebGLRendererResult {
  const stateRef = useRef<RendererState | null>(null);
  const grainSeedRef = useRef(Math.random() * 1000);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // draw() is called both synchronously (from the effect below) and later
  // from async image/LUT-load callbacks. Reading recipeAdjustment/imageUrl
  // through refs (kept current on every render, below) rather than closing
  // over the function arguments directly means a slow image decode — or a
  // context-restore replay — can't render with since-superseded values.
  const recipeAdjustmentRef = useRef(recipeAdjustment);
  recipeAdjustmentRef.current = recipeAdjustment;
  const imageUrlRef = useRef(imageUrl);
  imageUrlRef.current = imageUrl;

  function draw() {
    const state = stateRef.current;
    const canvas = canvasRef.current;
    const adjustment = recipeAdjustmentRef.current;
    if (!state || !canvas || !state.imageTexture || !state.lutTexture || !adjustment) {
      setIsReady(false);
      return;
    }
    const { gl, program, uniforms, imageTexture, lutTexture } = state;

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
    gl.uniform1f(uniforms.u_colorChromeStrength, Math.max(0, Math.min(1, adjustment.colorChromeStrength)));
    gl.uniform1f(uniforms.u_colorChromeFxBlueStrength, Math.max(0, Math.min(1, adjustment.colorChromeFxBlueStrength)));
    gl.uniform1f(
      uniforms.u_grainStrength,
      lerpEffectStrength(adjustment.grainStrength, GRAIN_STRENGTH_OFF, GRAIN_STRENGTH_WEAK, GRAIN_STRENGTH_STRONG),
    );
    gl.uniform1f(
      uniforms.u_grainSize,
      adjustment.grainSize ? (GRAIN_SIZE_SCALE[adjustment.grainSize] ?? DEFAULT_GRAIN_SIZE_SCALE) : DEFAULT_GRAIN_SIZE_SCALE,
    );
    gl.uniform1f(uniforms.u_grainSeed, grainSeedRef.current);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    setIsReady(true);
  }

  function loadImageTexture(url: string) {
    const canvas = canvasRef.current;
    if (!stateRef.current || !canvas) return;

    const image = new Image();
    image.onload = () => {
      const state = stateRef.current;
      const canvasEl = canvasRef.current;
      // Bail if the context was lost/torn down, or superseded by a newer
      // image, while this decode was in flight.
      if (!state || !canvasEl || imageUrlRef.current !== url) return;

      const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
      canvasEl.width = Math.round(image.naturalWidth * scale);
      canvasEl.height = Math.round(image.naturalHeight * scale);

      if (state.imageTexture) state.gl.deleteTexture(state.imageTexture);
      state.imageTexture = createImageTexture(state.gl, image);
      grainSeedRef.current = Math.random() * 1000;
      draw();
    };
    image.onerror = () => {
      if (imageUrlRef.current === url) setError("Failed to load the uploaded image.");
    };
    image.src = url;
  }

  function loadLutTextureFor(baseFilmSimulation: RecipeAdjustment["baseFilmSimulation"]) {
    const state = stateRef.current;
    if (!state) return;

    createLutTexture(state.gl, baseFilmSimulation)
      .then((texture) => {
        if (!stateRef.current || recipeAdjustmentRef.current?.baseFilmSimulation !== baseFilmSimulation) return;
        if (stateRef.current.lutTexture) stateRef.current.gl.deleteTexture(stateRef.current.lutTexture);
        stateRef.current.lutTexture = texture;
        draw();
      })
      .catch(() => {
        setError("Failed to load the film simulation LUT.");
      });
  }

  // Compile the program once per <canvas> element, and handle context loss.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // preserveDrawingBuffer is required for canvas.toBlob()/toDataURL() (used
    // by the "Save Image" export) to reliably read back the last-drawn frame
    // — without it the browser is allowed to clear the buffer after
    // compositing, and since this renderer only redraws on demand (no
    // continuous rAF loop), the buffer may already be stale/cleared by the
    // time the user clicks export.
    const contextOptions: WebGLContextAttributes = { preserveDrawingBuffer: true };
    const gl = (canvas.getContext("webgl2", contextOptions) ??
      canvas.getContext("webgl", contextOptions)) as GLContext | null;
    if (!gl) {
      setError("WebGL is not supported in this browser.");
      return;
    }

    function setUpProgram() {
      if (!gl) return;
      try {
        const program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
        createFullscreenQuad(gl, program);

        stateRef.current = {
          gl,
          program,
          uniforms: {
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
          },
          imageTexture: null,
          lutTexture: null,
        };
        setError(null);

        // Textures are lost along with the context (initial setup or after
        // a restore) — reload whatever the app currently has selected.
        const url = imageUrlRef.current;
        const sim = recipeAdjustmentRef.current?.baseFilmSimulation;
        if (url) loadImageTexture(url);
        if (sim) loadLutTextureFor(sim);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to initialize WebGL.");
      }
    }

    function handleContextLost(event: Event) {
      event.preventDefault(); // signals the browser we intend to recover
      stateRef.current = null;
      setIsReady(false);
      setError("Graphics context lost — attempting to recover...");
    }

    function handleContextRestored() {
      setUpProgram();
    }

    canvas.addEventListener("webglcontextlost", handleContextLost, false);
    canvas.addEventListener("webglcontextrestored", handleContextRestored, false);

    setUpProgram();

    return () => {
      canvas.removeEventListener("webglcontextlost", handleContextLost);
      canvas.removeEventListener("webglcontextrestored", handleContextRestored);
      const state = stateRef.current;
      if (!state) return;
      state.gl.deleteProgram(state.program);
      if (state.imageTexture) state.gl.deleteTexture(state.imageTexture);
      if (state.lutTexture) state.gl.deleteTexture(state.lutTexture);
      stateRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- setUpProgram/loadImageTexture/loadLutTextureFor read current state via refs, not via this effect's closure.
  }, [canvasRef]);

  // Upload the source image as a texture whenever it changes.
  useEffect(() => {
    if (!stateRef.current || !imageUrl) return;
    loadImageTexture(imageUrl);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl]);

  // Load the LUT texture whenever the recipe's base film simulation changes.
  useEffect(() => {
    if (!recipeAdjustment) return;
    loadLutTextureFor(recipeAdjustment.baseFilmSimulation);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recipeAdjustment?.baseFilmSimulation]);

  // Redraw whenever any other recipe adjustment field changes (uniform-only
  // update — no shader recompile, which is what keeps this fast).
  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recipeAdjustment]);

  return { isReady, error };
}
