/**
 * Service-layer placeholder for decoding Fujifilm .RAF files into raw pixel
 * data the WebGL engine can upload as a texture, alongside the existing
 * JPEG <input> path in hooks/useFileDrop.ts.
 *
 * Not implemented — this is scaffolding so the eventual integration has an
 * agreed call shape. Real implementation needs a WASM-compiled LibRaw build
 * (e.g. via libraw-wasm or a custom Emscripten build), invoked here to
 * decode the RAF's Bayer/X-Trans sensor data into an RGB buffer.
 */
export interface RawDecodeResult {
  width: number;
  height: number;
  /** Interleaved RGBA, one byte per channel — matches the shape createImageTexture expects. */
  pixels: Uint8ClampedArray;
}

export async function decodeRafFile(_file: File): Promise<RawDecodeResult> {
  throw new Error(
    "RAF decoding is not implemented yet — it requires a WASM-compiled LibRaw build. " +
      "See src/lib/raw/README.md for the integration plan.",
  );
}
