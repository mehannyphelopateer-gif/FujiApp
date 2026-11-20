#!/usr/bin/env node
/** F24 bounded source-metadata cross-check; no image interpretation or targets. */
import assert from 'node:assert/strict';
import {readFile,writeFile,open,stat} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readRafWbCoefficients} from './lib/raf-wb-coefficients.mjs';
const manifestPath='training/diagnostics/provia-runtime-wb-parser-manifest.json';
const evidencePath='training/diagnostics/provia-source-wb-probe.json';
const output='training/diagnostics/provia-runtime-wb-parser-verification.json';
const hash=async p=>{const h=createHash('sha256');for await(const b of createReadStream(p))h.update(b);return h.digest('hex');};
const bytesHash=b=>createHash('sha256').update(b).digest('hex');
if(process.argv[2]==='--prepare'){
 const paths=[evidencePath,'scripts/check-raf-wb-coefficients.mjs','scripts/lib/raf-wb-coefficients.mjs','scripts/test-raf-wb-coefficients.mjs'];
 const bindings=Object.fromEntries(await Promise.all(paths.map(async p=>[p,await hash(p)])));
 await writeFile(manifestPath,JSON.stringify({scope:'F24 source-only metadata crosscheck against independently verified Python/ExifTool; no unpack/export/targets/models/protected tiers. Auto policy restricted to qualified11, not applied here.',bindings},null,2)+'\n',{flag:'wx'});
 console.log(await hash(manifestPath));process.exit(0);
}
assert.equal(process.argv.length,3);assert.equal(await hash(manifestPath),process.argv[2]);
const m=JSON.parse(await readFile(manifestPath,'utf8'));
for(const[p,h]of Object.entries(m.bindings))assert.equal(await hash(p),h);
for(const p of [manifestPath,'scripts/check-raf-wb-coefficients.mjs','scripts/lib/raf-wb-coefficients.mjs'])assert.equal(bytesHash(execFileSync('git',['show',`HEAD:${p}`])),await hash(p),'Uncommitted source/manifest');
const e=JSON.parse(await readFile(evidencePath,'utf8'));assert(e.crosschecks_passed);assert.equal(e.rows.length,24);
const rows=[];
for(const r of e.rows){
 const path=`calibration-input/${r.shoot}/${r.name}`;assert.equal(await hash(path),r.raf_sha256);
 const handle=await open(path,'r');let actual;
 try{
  const size=(await stat(path)).size;
  const source={size,slice:(start,end)=>({arrayBuffer:async()=>{const b=Buffer.alloc(end-start);const n=await handle.read(b,0,b.length,start);assert.equal(n.bytesRead,b.length);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);}})};
  actual=await readRafWbCoefficients(source);
 }finally{await handle.close();}
 for(const[tag,key]of [['0xf00d','autoRgbg'],['0xf00e','asShotCamMul']]){
  const records=r.records.filter(x=>x.tag===tag&&x.supported);assert(records.length>0);
  const grb=records[0].values;assert.deepEqual(actual[key],[grb[1],grb[0],grb[2],tag==='0xf00d'?grb[0]:0]);
 }
 assert(r.as_shot_records_agree);assert(r.archived_cam_mul_matching_as_shot_tags.includes('0xf00e'));
 assert.equal(await hash(path),r.raf_sha256);
 rows.push({name:r.name,shoot:r.shoot,source_raf_sha256:r.raf_sha256,source_wb_mode:r.source_wb_mode,...actual,passed:true});
 console.log(`${rows.length}/24 ${r.name} exact ${actual.bytesRead} metadata bytes`);
}
await writeFile(output,JSON.stringify({passed:true,manifest_sha256:process.argv[2],scope:m.scope,rows},null,2)+'\n',{flag:'wx'});
