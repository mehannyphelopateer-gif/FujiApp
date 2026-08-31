# Third-party assets

## Film simulation LUTs (`public/luts/`)

11 of the 14 film-simulation Hald CLUT PNGs — `astia`, `classic-chrome`,
`classic-negative`, `eterna`, `eterna-bleach-bypass`, `nostalgic-neg`,
`pro-neg-hi`, `pro-neg-std`, `provia`, `reala-ace`, `velvia` — are adapted
from:

**Fujifilm Camera Profiles** by abpy
https://github.com/abpy/FujifilmCameraProfiles
Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

These are Adobe Camera Raw/Lightroom's own camera-matching profiles for
Fuji X-series cameras (based on an X-Trans IV body per the source repo),
not derived from Fujifilm's own conversion software — an approximation,
not an exact reproduction of Fuji's in-camera color science. Converted
from the source's level-6 (216×216, 36 levels/channel) Hald CLUT format to
this app's level-8 (512×512, 64 levels/channel) format via
`scripts/convert-third-party-luts.mjs` (trilinear resampling, no other
modification).

`acros`, `monochrome`, and `sepia` are not covered by this source and
remain the original placeholder LUTs from `scripts/generate-placeholder-luts.mjs`
pending a real replacement.

**License terms that apply to these 11 files specifically** (not the rest
of this app): non-commercial use only, share-alike (any redistributed
modification of these specific files must carry the same CC BY-NC-SA 4.0
license), attribution required. See the license text at the link above
for the complete terms.

See `~/.claude/plans/indexed-inventing-wren.md` (or git history) for the
self-calibration approach planned to eventually replace these with LUTs
derived directly from this app's own camera, once real calibration photos
are available — see that plan for why: these are a real improvement over
the hand-guessed placeholders, but calibrating against the actual camera's
own RAW-conversion engine remains the more accurate long-term goal.
