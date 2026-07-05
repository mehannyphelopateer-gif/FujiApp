# RAW / .RAF

`rawService.ts` defines the intended call shape (`decodeRafFile(file) -> { width, height, pixels }`)
but is unimplemented — it throws until a WASM-compiled LibRaw build is wired in.

Integration plan: compile LibRaw to WebAssembly (e.g. `libraw-wasm` or a custom Emscripten
build), decode a `.RAF` file's Bayer/X-Trans sensor data into an interleaved RGBA buffer matching
`RawDecodeResult`, and feed that into `engine/webgl/glUtils.ts`'s texture upload path as an
alternative source to the JPEG `<img>`/`<canvas>` path used by `hooks/useFileDrop.ts`.
