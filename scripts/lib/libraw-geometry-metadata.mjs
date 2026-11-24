/** Numeric geometry only. No pixel access and no expected-value substitution. */
export const SIZE_KEYS = Object.freeze(['width','height','raw_width','raw_height','top_margin','left_margin','flip','iwidth','iheight','raw_pitch','pixel_aspect','raw_aspect']);
export const CROP_KEYS = Object.freeze(['cleft','ctop','cwidth','cheight']);
export function selectGeometry(meta) {
  if (!meta || typeof meta !== 'object') throw new Error('Missing decoder metadata');
  const sizes = {};
  for (const k of SIZE_KEYS) {
    const v = meta[k];
    if (v == null) sizes[k] = null;
    else { if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`Non-numeric ${k}`); sizes[k] = v; }
  }
  let crops = null;
  if (meta.raw_inset_crops != null) {
    if (!Array.isArray(meta.raw_inset_crops) || meta.raw_inset_crops.length > 4) throw new Error('Invalid inset crops');
    crops = meta.raw_inset_crops.map(c => Object.fromEntries(CROP_KEYS.map(k => {
      if (typeof c?.[k] !== 'number' || !Number.isFinite(c[k])) throw new Error(`Invalid crop ${k}`);
      return [k,c[k]];
    })));
  }
  return {...sizes, raw_inset_crops:crops,
    not_returned:[...SIZE_KEYS.filter(k => sizes[k] === null), ...(crops === null ? ['raw_inset_crops'] : [])]};
}
