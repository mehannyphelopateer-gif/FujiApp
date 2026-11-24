/** Cross-language synthetic codec QA only. No photo/media/decoder/model access. */
import assert from 'node:assert/strict';
import { readFile,writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import * as N from './lib/fjnrgb-profile.mjs';
import * as V from './lib/fjlrg-v2.mjs';
import { makeFixture } from './lib/fjlrg-v2-fixtures.mjs';
const dir=process.argv[2];assert(dir,'Synthetic fixture directory required');
const bytes=new Uint8Array(await readFile(resolve(dir,'python.fjnrgb')));
const provenance=V.parseStrictJson(await readFile(resolve(dir,'python.fjnrgb.json'),'utf8'));
const expected={expectedSourceSha256:'a'.repeat(64),expectedWasmSha256:'b'.repeat(64)};
let checks=0;
const reject=async(b,p)=>{await assert.rejects(()=>N.decodeFjnrgbProfile(b,p,expected));checks++;};
const frame=await N.decodeFjnrgbProfile(bytes,provenance,expected);checks++;
const options=V.serializeOptions(N.canonicalOptions());
const encoded=N.encodeFjnrgbProfile(frame,{optionsSha256:await V.sha256Hex(new TextEncoder().encode(options))});
assert.deepEqual(encoded,bytes);checks++;
const built=await N.buildProvenance(encoded,{sourceRafSha256:expected.expectedSourceSha256,wasmSha256:expected.expectedWasmSha256,captureDynamicRange:200,captureDynamicRangeSource:'synthetic fixture only',decodedNativeSize:[frame.width,frame.height]});
assert.deepEqual(built,provenance);checks++;
await assert.rejects(()=>V.decodeFjlrgV2(bytes,provenance,{...expected,expectedContract:'v2-linear-asshot'}));checks++;
for(const contract of ['v2-linear-asshot','v2-linear-autowb']){
 const f=await makeFixture({contract});await reject(f.bytes,f.provenance);
}
for(const off of [0,8,10,12,16,20,22,24,28,32,34,36,38,40,44,48,52,56]){
 const bad=new Uint8Array(bytes);bad[off]^=1;
 const p=structuredClone(provenance);p.container_sha256=await V.sha256Hex(bad);
 await reject(bad,p);
}
for(const [key,value] of Object.entries({schema:'wrong',input_contract:'v2-linear-autowb',source_wb_mode:0,source_kelvin:5600,source_wb_fine_tune_raw:[0,0],profile_key:'4080/5600',profile_manifest_sha256:'0'.repeat(64),policy_basis:'fallback',parameters_are_declared:false,capture_dynamic_range:NaN,decoded_native_size:[0,3],wb_source:{kind:'fuji_f00d',levels_rgbg:[620,302,455,302]},primitive_reports:{},decoder_version:'other'})){
 const p=structuredClone(provenance);p[key]=value;await reject(bytes,p);
}
for(const key of Object.keys(provenance)){
 const p=structuredClone(provenance);delete p[key];await reject(bytes,p);
}
const extra=structuredClone(provenance);extra.target_gain=[620,302,455];await reject(bytes,extra);
for(const field of ['useCameraWb','gamm','userMul']){
 const p=structuredClone(provenance);const opts=N.canonicalOptions();
 opts[field]=field==='useCameraWb'?true:field==='gamm'?[1,1]:[636,302,414,302];
 p.options_json=V.serializeOptions(opts);p.declared_options=opts;p.options_sha256=await V.sha256Hex(new TextEncoder().encode(p.options_json));
 const b=new Uint8Array(bytes);for(let i=0;i<8;i++)b[56+i]=Number.parseInt(p.options_sha256.slice(i*2,i*2+2),16);
 p.container_sha256=await V.sha256Hex(b);await reject(b,p);
}
await assert.rejects(()=>N.decodeFjnrgbProfile(bytes,provenance,{}));checks++;
await assert.rejects(()=>N.decodeFjnrgbProfile(bytes,provenance,{...expected,expectedSourceSha256:'A'.repeat(64)}));checks++;
const numericClaim=structuredClone(provenance);numericClaim.parameters_are_declared=1;await reject(bytes,numericClaim);
assert.throws(()=>N.encodeFjnrgbProfile({width:1537,height:1,data:new Uint16Array(1537*3)},{optionsSha256:provenance.options_sha256}));checks++;
const duplicate=structuredClone(provenance);
duplicate.options_json=provenance.options_json.slice(0,-1)+',"useCameraWb":false}';
duplicate.options_sha256=await V.sha256Hex(new TextEncoder().encode(duplicate.options_json));
const duplicateBytes=new Uint8Array(bytes);
for(let i=0;i<8;i++)duplicateBytes[56+i]=Number.parseInt(duplicate.options_sha256.slice(i*2,i*2+2),16);
duplicate.container_sha256=await V.sha256Hex(duplicateBytes);
await reject(duplicateBytes,duplicate);
await writeFile(resolve(dir,'javascript.fjnrgb'),encoded);
await writeFile(resolve(dir,'javascript.fjnrgb.json'),JSON.stringify(built));
const report={synthetic_only:true,passed:true,checks,container_sha256:await V.sha256Hex(encoded),python_javascript_byte_identical:true,provenance_identical:true};
await writeFile(resolve(dir,'javascript-report.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
