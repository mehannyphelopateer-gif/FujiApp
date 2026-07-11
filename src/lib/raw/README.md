# RAW / .RAF

`rawService.ts` extracts the full-size JPEG preview every `.RAF` embeds alongside its raw sensor
data, rather than decoding the actual Bayer/X-Trans sensor data (that would need a WASM-compiled
LibRaw build — a large dependency, out of scope for a recipe *previewer*). The extracted JPEG is
fed through the same `hooks/useFileDrop.ts` → `AppStateContext` path as a native JPEG upload, so
it gets the normal WebGL preview pipeline and EXIF-based detected-settings for free.

`extractRafPreviewJpeg(file)` reads the embedded JPEG's byte offset and length directly from two
fixed big-endian uint32 fields in the RAF header (`0x54`/`0x58`) — confirmed against exiftool's
`FujiFilm.pm` source and libopenraw's RAF format docs, and verified against a real sample file
(2026-07). An earlier version scanned for JPEG SOI/EOI marker bytes instead, which is unreliable:
the embedded preview JPEG typically carries its own nested EXIF thumbnail with its own SOI/EOI
markers, so a naive scan can grab the wrong one.
