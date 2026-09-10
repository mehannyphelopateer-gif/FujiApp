# Phase 3 linear RGB calibration container

Owner: Codex. This is the browser-side half of the Phase 3 data contract.

`decodePhase3LinearRaf()` in `src/lib/raw/rawService.ts` exports a neutral
RAF decode as a `.fjlrg` file with MIME type `application/x-fujiapp-linear-rgb`.
It is calibration-only; normal Preview continues to use its displayable JPEG
path until a Phase 3 model passes validation and is deliberately integrated.

## Decode settings

- Full-resolution X-Trans Markesteijn demosaic (`halfSize: false`, `userQual: 3`)
- 16-bit output (`outputBps: 16`)
- Camera matrix to linear sRGB primaries (`outputColor: 1`, `useCameraMatrix: 1`)
- Linear transfer curve (`gamm: [1, 1]`)
- As-shot camera white balance (`useCameraWb: true`, `useAutoWb: false`)
- No auto-brightening, highlight recovery, or FBDD denoise
- Area-average downsampled in linear space to a 1536-pixel long edge

## Binary layout

The file has a 32-byte little-endian header followed by tightly packed,
interleaved RGB `uint16` samples in little-endian order.

| Offset | Type | Value |
| --- | --- | --- |
| 0 | 8 ASCII bytes | `FJLRGB16` |
| 8 | uint16 | format version (`1`) |
| 10 | uint16 | header bytes (`32`) |
| 12 | uint32 | width |
| 16 | uint32 | height |
| 20 | uint16 | channels (`3`) |
| 22 | uint16 | bits per channel (`16`) |
| 24 | uint32 | payload byte count (`width * height * 3 * 2`) |
| 28 | uint32 | flags: bit 0 = linear transfer, bit 1 = sRGB primaries |
| 32 | uint16[] | RGBRGB… payload |

The Phase 3 browser filename is `browser-phase3-linear.fjlrg` in each shoot
folder. The batch exporter deliberately covers only Shoots 127–386 (training
and monitoring). It never reads or writes the locked final-test range.
