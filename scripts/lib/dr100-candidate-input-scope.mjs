import assert from 'node:assert/strict';
export const NAMES=Object.freeze(Array.from({length:9},(_,i)=>`DSCF${1424+2*i}.RAF`));
export const BASE='calibration-input/phase4-dr100eval-attempt1-excluded';
export const OUTPUT='training/checkpoints/dr100-candidate-inputs';
export function validateInputScope(m) {
  assert.equal(m.format,'dr100-candidate-G-input-v1');assert.equal(m.training_admission,false);
  assert.match(m.wasm_sha256,/^[0-9a-f]{64}$/);
  assert.equal(m.wasm_sha256,m.bindings['node_modules/libraw-wasm/dist/libraw.wasm']);
  assert.equal(m.output_dir,OUTPUT);assert.deepEqual(m.scenes.map(s=>s.name),NAMES);
  assert.equal(m.canary_names.length,2);assert.equal(new Set(m.canary_names).size,2);
  assert(m.canary_names.every(n=>NAMES.includes(n)));
  for (const s of m.scenes) {
    assert.equal(s.path,`${BASE}/${s.name}`);assert.match(s.raf_sha256,/^[0-9a-f]{64}$/);
    assert.deepEqual(s.source_wb_fine_tune_raw,[0,0]);assert([1,4080].includes(s.source_wb_mode));
    assert.equal(s.capture_dynamic_range,100);assert.deepEqual(s.options.gamm,[1,1,0,0,0,0]);
    assert.equal(s.options.outputBps,16);assert.equal(s.options.useCameraWb,true);
  }
  assert.deepEqual(new Set(m.canary_names.map(n=>m.scenes.find(s=>s.name===n).source_wb_mode)),new Set([1,4080]));
  return m;
}
