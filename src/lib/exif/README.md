# EXIF

- `parseFujiMakerNotes.ts` — `extractDetectedSettings(file)` reads a Fuji JPEG's EXIF MakerNotes
  into a `DetectedSettings` object: the film simulation, white balance, and tone settings the
  camera actually baked into the image. Returns `null` for non-Fuji cameras or stripped EXIF.
- `sensorGenerations.ts` — `mapCameraModelToSensorGeneration(model)` maps a camera model string
  (e.g. `"X-T5"`) to its X-Trans sensor generation (e.g. `"X-Trans V"`), used to filter the
  recipe grid to `Recipe.compatibleSensors`.

**Verified against a real file (2026-07)** — `exifreader` has no Fujifilm-specific MakerNote
support at all (only Canon/Pentax; confirmed by reading its source), so `tags.Fujifilm` is always
`undefined` and an earlier version of this function always returned `null`. It now parses the raw
`tags.exif.MakerNote` binary IFD directly — tag IDs and value tables are transcribed from
exiftool's `FujiFilm.pm` and cross-checked against a real MakerNote block decoded byte-by-byte
(an X-Pro1 sample file's embedded RAF preview JPEG). See the file's own top comment for the wire
format and sources.

See `src/lib/recipes/neutralize.ts` for how `DetectedSettings` becomes a `RecipeAdjustment`
(the delta actually sent to the shader).
