// DEFAULT/FALLBACK values — no real Phase 3 calibration shoot has been run
// yet, so these exactly reproduce today's hand-picked
// useWebGLRenderer.ts grain constants, making the switch to importing this
// file a zero-behavior-change no-op until real data replaces it.
//
// Once a Phase 3 calibration shoot exists, run
// `node scripts/derive-grain-stats.mjs` to OVERWRITE this file with real
// measured noise statistics — see
// ~/.claude/plans/indexed-inventing-wren.md's Phase 3 section.

export const GRAIN_STRENGTH_WEAK = 0.035;
export const GRAIN_STRENGTH_STRONG = 0.08;
export const GRAIN_SIZE_SCALE: Record<"Small" | "Large", number> = { Small: 0.9, Large: 0.35 };
