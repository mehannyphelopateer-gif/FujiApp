/** Synthetic protocol fixtures only; the hashes do not identify user media. */
import { encodeFjlrgV2, serializeOptions, sha256Hex } from "./fjlrg-v2.mjs";
export function fixtureOptions(contract, wb) {
  const options = { halfSize: false, outputBps: 16, outputColor: 1, gamm: [1, 1, 0, 0, 0, 0], noAutoBright: true,
    adjustMaximumThr: 0, useCameraMatrix: 1, useCameraWb: contract === "v2-linear-asshot", useAutoWb: false,
    highlight: 0, userQual: 3, useFujiRotate: -1, fbddNoiserd: 0 };
  if (contract === "v2-linear-autowb") options.userMul = wb;
  return options;
}

export async function makeFixture({ contract = "v2-linear-asshot", width = 4, height = 3,
  levels = contract === "v2-linear-asshot" ? [628, 302, 445, 0] : [518, 302, 570, 302], optionsJson } = {}) {
  const pixels = Uint16Array.from({ length: width * height * 3 }, (_, i) => (i * 4051 + 0x0102) % 65536);
  pixels[0] = 0x0102; pixels[1] = 0; pixels[2] = 65535;
  const options = fixtureOptions(contract, levels);
  const text = optionsJson ?? serializeOptions(options);
  const digest = await sha256Hex(new TextEncoder().encode(text));
  const bytes = encodeFjlrgV2({ width, height, data: pixels }, { inputContract: contract, wbLevelsRgbg: levels, optionsSha256: digest });
  const provenance = { schema: "fujiapp.raw-input.v2", input_contract: contract,
    container_sha256: await sha256Hex(bytes), source_raf_sha256: "ab".repeat(32), wasm_sha256: "cd".repeat(32),
    decoder_package: "libraw-wasm", decoder_version: "1.6.0", libraw_build_version: "0.22.1",
    options_json: text, options_sha256: digest, declared_options: JSON.parse(text),
    wb_source: { kind: contract === "v2-linear-asshot" ? "cam_mul" : "fuji_f00d", levels_rgbg: levels },
    source_wb_mode: contract === "v2-linear-asshot" ? 4080 : 0, source_wb_fine_tune_raw: [20, -60],
    source_kelvin: contract === "v2-linear-asshot" ? 6600 : null,
    capture_dynamic_range: 400, capture_dynamic_range_source: "synthetic MakerNote fixture",
    decoded_native_size: [width, height], parameters_are_declared: true };
  return { bytes, provenance, pixels, contract, width, height };
}
