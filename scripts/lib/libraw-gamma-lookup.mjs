/** Fixed LibRaw0.22.1 gamma_curve mode2, full-scale white, no fitted values. */
export function librawGammaLookup(power = 0.45, toe = 4.5) {
  let lo = 0, hi = 1, pivot = 0;
  for (let i = 0; i < 48; i++) {
    pivot = (lo + hi) / 2;
    if ((Math.pow(pivot / toe, -power) - 1) / power - 1 / pivot > -1) hi = pivot;
    else lo = pivot;
  }
  const join = pivot / toe, shoulder = pivot * (1 / power - 1);
  const table = new Uint16Array(65536);
  for (let i = 0; i < table.length; i++) {
    const x = i / 65536;
    table[i] = Math.min(65535, Math.max(0, Math.floor(65536 * (x < join ? x * toe : Math.pow(x, power) * (1 + shoulder) - shoulder))));
  }
  return table;
}
