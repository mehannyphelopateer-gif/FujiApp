# Third-party assets

## Film simulation LUTs (`public/luts/`)

None currently — all 14 film-simulation Hald CLUT PNGs are now derived
directly from this app's own camera via
`scripts/derive-luts-from-calibration.mjs` (real calibration photos, fitted
per the methodology in `~/.claude/plans/indexed-inventing-wren.md`), so
none of them carry a license restriction.

### Historical note

8 of the 14 (`astia`, `classic-negative`, `eterna`,
`eterna-bleach-bypass`, `nostalgic-neg`, `pro-neg-hi`, `pro-neg-std`,
`reala-ace`) were previously adapted from:

**Fujifilm Camera Profiles** by abpy
https://github.com/abpy/FujifilmCameraProfiles
Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

These were Adobe Camera Raw/Lightroom's own camera-matching profiles for
Fuji X-series cameras (based on an X-Trans IV body per the source repo),
not derived from Fujifilm's own conversion software — an approximation,
not an exact reproduction of Fuji's in-camera color science, used as an
interim stand-in (via `scripts/convert-third-party-luts.mjs`, trilinear
resampling from the source's level-6 format to this app's level-8 format)
before real calibration photos were available. None of that source's
output is shipped in this app anymore — see git history if the CC
BY-NC-SA 4.0 terms (non-commercial, share-alike, attribution) ever need
checking against a past version of `public/luts/`.
