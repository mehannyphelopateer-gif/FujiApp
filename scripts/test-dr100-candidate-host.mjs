import assert from 'node:assert/strict';
import { candidate, NAMES, BASE } from './read-dr100-candidate-metadata.mjs';
const s={scenes:NAMES.map(name=>({name,path:`${BASE}/${name}`,raf_sha256:'a'.repeat(64)}))};
assert.equal(candidate(s,NAMES[0]).name,NAMES[0]);
for (const n of ['DSCF1425.RAF','DSCF1443.RAF','../DSCF1424.RAF','finalHeldOut'])
  assert.throws(()=>candidate(s,n),/allowlist/);
assert.throws(()=>candidate({scenes:s.scenes.slice(0,8)},NAMES[0]),/scope/);
const escaped=structuredClone(s);escaped.scenes[0].path='calibration-input/Shoot 744/DSCF1443.RAF';
assert.throws(()=>candidate(escaped,NAMES[0]),/path/);
console.log('Scoped candidate host guards pass without source I/O.');
