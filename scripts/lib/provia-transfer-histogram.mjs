/** Exact native-code transfer arithmetic; no image I/O or fitted parameters. */
export const MIN_LINEAR_CODE = 656;
export const MAX_LINEAR_CODE = 62258;
export function histogramSummary(histogram, perChannelN, ineligibleCount) {
  if (!(histogram instanceof Uint32Array) || histogram.length !== 65536) throw new Error('Expected 65536-bin histogram');
  const sparse = []; let n = 0, sum = 0, max = null;
  for (let e = 0; e < histogram.length; e++) if (histogram[e]) {
    const count = histogram[e]; sparse.push([e, count]); n += count; sum += e * count; max = e;
  }
  if (!Array.isArray(perChannelN) || perChannelN.length !== 3 || !perChannelN.every(Number.isSafeInteger) || perChannelN.some(x => x < 0) || perChannelN.reduce((a,b)=>a+b,0) !== n) throw new Error('Channel count mismatch');
  if (!Number.isSafeInteger(ineligibleCount) || ineligibleCount < 0) throw new Error('Invalid ineligible count');
  let p99 = null;
  if (n) {
    const q = .99 * (n - 1), lo = Math.floor(q), hi = Math.ceil(q);
    let cumulative = 0, loValue, hiValue;
    for (const [e, count] of sparse) {
      cumulative += count;
      if (loValue === undefined && cumulative > lo) loValue = e;
      if (cumulative > hi) { hiValue = e; break; }
    }
    p99 = loValue + (hiValue - loValue) * (q - lo);
  }
  return { n, mean_abs_codes: n ? sum / n : null, p99_abs_codes: p99, max_abs_codes: max,
    exact_fraction: n ? histogram[0] / n : null, histogram: sparse, per_channel_n: perChannelN, ineligible_count: ineligibleCount,
    passed: n >= 1000 && sum / n <= 2 && p99 <= 3 };
}
export function measureNativeTransfer(old, linear, table) {
  if (!(old instanceof Uint16Array) || !(linear instanceof Uint16Array) || old.length !== linear.length || old.length % 3 || table.length !== 65536) throw new Error('Native geometry/type mismatch');
  const histogram = new Uint32Array(65536), perChannel = [0,0,0]; let ineligible = 0;
  for (let i=0; i<linear.length; i++) {
    const v = linear[i];
    if (v < MIN_LINEAR_CODE || v > MAX_LINEAR_CODE) { ineligible++; continue; }
    histogram[Math.abs(old[i] - table[v])]++; perChannel[i%3]++;
  }
  return histogramSummary(histogram, perChannel, ineligible);
}
