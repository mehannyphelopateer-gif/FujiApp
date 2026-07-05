# EXIF

- `parseFujiMakerNotes.ts` — `extractDetectedSettings(file)` reads a Fuji JPEG's EXIF MakerNotes
  (via `exifreader`) into a `DetectedSettings` object: the film simulation, white balance, and
  tone settings the camera actually baked into the image. Returns `null` for non-Fuji cameras or
  stripped EXIF.
- `sensorGenerations.ts` — `mapCameraModelToSensorGeneration(model)` maps a camera model string
  (e.g. `"X-T5"`) to its X-Trans sensor generation (e.g. `"X-Trans V"`), used to filter the
  recipe grid to `Recipe.compatibleSensors`.

**Unverified against real files** — this app was built without a runtime available to test with
(no Node.js in the dev environment). The tag names and raw-value scaling constants in
`parseFujiMakerNotes.ts` (division by 20 for `WhiteBalanceFineTune`, division by 2 for tone
values) are best-effort, called out inline. Before trusting this: run
`ExifReader.load(file, { expanded: true })` against a handful of real Fuji JPEGs from different
bodies/firmware, log the raw `tags.Fujifilm` object, and correct anything that doesn't match.

See `src/lib/recipes/neutralize.ts` for how `DetectedSettings` becomes a `RecipeAdjustment`
(the delta actually sent to the shader).
