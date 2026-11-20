/** Exercise the actual browser instrumentation against brute-force fixtures. */
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
const source = readFileSync(new URL("./probe-provia-native-clipping.mjs", import.meta.url), "utf8");
const start = source.indexOf("const n = image.width * image.height, radius = 6;");
const end = source.indexOf("// Fixed native samples", start);
assert(start >= 0 && end > start);
const instrument = new Function("image", source.slice(start, end) + "return nearWhite;");
for (const colors of [3, 4]) for (const [py, px] of [[9, 8], [0, 0], [18, 16]]) {
  const image = { width: 17, height: 19, colors, data: new Uint16Array(17 * 19 * colors) };
  if (colors === 4) for (let i = 0; i < 17 * 19; i++) image.data[i * 4 + 3] = 65535;
  assert(instrument(image).every(v => v === 0), "Fourth channel is not a clipped RGB sample");
  image.data[(py * 17 + px) * colors + 1] = 65535;
  const before = new Uint16Array(image.data), out = instrument(image);
  assert.deepEqual(image.data, before, "Instrumentation must be read-only");
  for (let y = 0; y < 19; y++) for (let x = 0; x < 17; x++)
    assert.equal(out[y * 17 + x], Number(Math.abs(y - py) <= 6 && Math.abs(x - px) <= 6));
}
console.log("Native-radius6 instrumentation: RGB/RGBG, edges and read-only fixtures PASS");
