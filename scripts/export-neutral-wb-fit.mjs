#!/usr/bin/env node
/** Frozen FIT-only WB qualification; old geometry, no targets/models/app. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {createServer} from 'node:http';
import {createReadStream} from 'node:fs';
import {readFile,writeFile,mkdir,stat} from 'node:fs/promises';
import {resolve,relative,basename,extname} from 'node:path';
import {chromium} from 'playwright';
import * as V from './lib/fjlrg-v2.mjs';
import * as P from './lib/fjnrgb-profile.mjs';
import {armOptions,normalizeDecoderResult} from './lib/neutral-wb-artifacts.mjs';
const root=process.cwd(),manifest='training/diagnostics/neutral-wb-fit-export-manifest.json';
const sha=b=>createHash('sha256').update(b).digest('hex');
const hashFile=async p=>{const h=createHash('sha256');for await(const b of createReadStream(p))h.update(b);return h.digest('hex');};
const json=async p=>JSON.parse(await readFile(p,'utf8'));
const exists=async p=>{try{await stat(p);return true;}catch(e){if(e.code==='ENOENT')return false;throw e;}};
const writeOnce=async(p,obj)=>{await writeFile(p,JSON.stringify(obj,null,2)+'\n',{flag:'wx'});return hashFile(p);};
const expected=process.argv[2],canaryOnly=process.argv.includes('--canaries');
assert(/^[0-9a-f]{64}$/.test(expected)&&process.argv.length===(canaryOnly?4:3),'Usage: manifestSHA [--canaries]');
assert.equal(await hashFile(manifest),expected);const m=await json(manifest);
assert.equal(sha(execFileSync('git',['show',`HEAD:${manifest}`])),expected);
for(const[p,h]of Object.entries(m.bindings)){
 assert.equal(await hashFile(p),h,`Binding drift ${p}`);
 if(!p.startsWith('node_modules/')&&!p.startsWith('training/checkpoints/'))assert.equal(sha(execFileSync('git',['show',`HEAD:${p}`])),h,`Uncommitted ${p}`);
}
assert.equal(m.scenes.length,455);assert.equal(new Set(m.scenes.map(s=>s.name)).size,455);
assert.equal(m.scenes.filter(s=>s.requires_new_decode).length,441);
const out=resolve(root,m.output_dir);assert.equal(out,resolve(root,'training/checkpoints/neutral-wb-fit-export'));
const sources=new Map();
for(const s of m.scenes){assert.equal(s.role,'fit');assert.match(s.name,/^DSCF\d{4}\.RAF$/);assert.match(s.shoot,/^Shoot \d+$/);assert(['W','P','G'].includes(s.kind));sources.set(s.name,resolve(root,'calibration-input',s.shoot,s.name));}
await mkdir(out,{recursive:true});const job=resolve(out,'job.json');
if(await exists(job))assert.deepEqual(await json(job),{manifest_sha256:expected});else await writeOnce(job,{manifest_sha256:expected});
if(!canaryOnly){
 const proof=await json('training/diagnostics/neutral-wb-fit-canary-verification.json');
 assert.equal(proof.verified,true);assert.equal(proof.canary_only,true);assert.equal(proof.completed_fit_rows,16);assert.equal(proof.manifest_sha256,expected);
 for(const[name,h]of Object.entries(proof.scene_seal_sha256)){const s=m.scenes.find(x=>x.name===name);assert(s);assert.equal(await hashFile(resolve(out,s.shoot,'scene-seal.json')),h);}
}
function safeArtifact(p){assert(p.startsWith('training/checkpoints/'));assert(!p.split('/').includes('..'));assert(['provia-input-v2-train-g','provia-input-v2-wb-pilot','profile-rgb-sentinels','neutral-wb-fit-export'].includes(p.split('/')[2]));return resolve(root,p);}
async function originalG(s){
 const p=safeArtifact(s.G.file);assert.equal(await hashFile(p),s.G.sha256);assert.equal(await hashFile(p+'.json'),s.G.sidecar_sha256);
 const bytes=await readFile(p),frame=await V.decodeFjlrgV2(new Uint8Array(bytes),await json(p+'.json'),{expectedContract:'v2-linear-asshot',expectedSourceSha256:s.raf_sha256,expectedWasmSha256:m.wasm_sha256});
 assert.deepEqual([frame.width,frame.height],s.expected_working_size);
 const payload=Buffer.from(frame.data.buffer,frame.data.byteOffset,frame.data.byteLength);assert.equal(sha(payload),s.G.working_payload_sha256);
 const strips=await readFile(safeArtifact(s.G.strips.file));assert.equal(sha(strips),s.G.strips.sha256);
 return {payload,strips};
}
async function validateSaved(s,arm,r,g){
 const old=s.reuse?.source_record;
 const prior=old&&s.kind==='W'?old.arms[{C:'control',N:'W',R:'W-repeat'}[arm]]:old&&s.kind==='P'?old.arms[{C:'C',N:'P',R:'R'}[arm]]:null;
 const expectedFile=prior?prior.file:`training/checkpoints/neutral-wb-fit-export/${s.shoot}/${arm}${arm==='C'?'-control.u16le':s.kind==='W'?'-neutral.fjlrg':'-neutral.fjnrgb'}`;
 const expectedStrips=prior?(s.kind==='W'?prior.strips.file:prior.strips_file):`training/checkpoints/neutral-wb-fit-export/${s.shoot}/${arm}-native-strips.u16le`;
 assert.equal(r.file,expectedFile,'Artifact outside exact scene allowlist');assert.equal(r.strips_file,expectedStrips,'Strips outside exact scene allowlist');
 const bytes=await readFile(safeArtifact(r.file)),strips=await readFile(safeArtifact(r.strips_file));
 assert.equal(sha(bytes),r.sha256);assert.equal(sha(strips),r.strips_sha256);
 assert.deepEqual(r.native_size,s.expected_native_size);assert.deepEqual([r.width,r.height],s.expected_working_size);
 assert.deepEqual(r.source_coefficients,{autoRgbg:s.auto_rgbg,asShotCamMul:s.cam_mul});
 assert.match(r.native_sha256,/^[0-9a-f]{64}$/);assert.equal(r.options_json,V.serializeOptions(armOptions(s,arm)));assert.equal(sha(r.options_json),r.options_sha256);
 const rows=[];for(let i=64;i<s.expected_native_size[1];i+=128)rows.push(i);
 assert.deepEqual(r.strip_rows,rows);assert.equal(strips.length,rows.length*s.expected_native_size[0]*6);
 if(s.kind==='W'){assert.equal(r.returned_metadata,null);assert.match(r.metadata_note,/not returned/i);}
 else{assert.equal(r.returned_metadata.camera_model,'X100VI');assert.equal(r.returned_metadata.fujiWB,s.source_wb_mode);assert.deepEqual(r.returned_metadata.cam_mul,s.cam_mul);}
 let payload;
 if(arm==='C'){
  assert.equal(r.contract,null);assert(!r.sidecar_sha256);payload=bytes;
  assert.deepEqual(payload,g.payload);assert.deepEqual(strips,g.strips);assert.equal(r.native_sha256,s.G.native_sha256);
 }else{
  const sidecar=await readFile(safeArtifact(r.file)+'.json');assert.equal(sha(sidecar),r.sidecar_sha256);const prov=JSON.parse(sidecar);
  assert.equal(r.contract,s.contract);assert.equal(prov.options_json,r.options_json);assert.equal(prov.capture_dynamic_range,s.capture_dynamic_range);assert.deepEqual(prov.decoded_native_size,s.expected_native_size);
  const opts={expectedSourceSha256:s.raf_sha256,expectedWasmSha256:m.wasm_sha256};
  const frame=s.kind==='W'?await V.decodeFjlrgV2(new Uint8Array(bytes),prov,{...opts,expectedContract:'v2-linear-autowb'}):await P.decodeFjnrgbProfile(new Uint8Array(bytes),prov,opts);
  assert.deepEqual([frame.width,frame.height],s.expected_working_size);payload=Buffer.from(frame.data.buffer,frame.data.byteOffset,frame.data.byteLength);
 }
 assert.equal(payload.length,s.expected_working_size[0]*s.expected_working_size[1]*6);assert.equal(sha(payload),r.working_payload_sha256);
 return {bytes,strips};
}
function repeatEqual(n,r){for(const k of ['native_sha256','working_payload_sha256','sha256','strips_sha256','options_json','options_sha256','sidecar_sha256'])assert.equal(n[k],r[k],`Repeat mismatch ${k}`);}
async function validateSeal(s,seal,g){
 assert.equal(seal.manifest_sha256,expected);assert.equal(seal.name,s.name);assert.equal(seal.role,'fit');assert.equal(seal.kind,s.kind);assert.equal(seal.source_raf_sha256,s.raf_sha256);
 if(s.kind==='G'){assert.deepEqual(seal.identity,s.G);assert(!seal.arms);return;}
 assert.deepEqual(Object.keys(seal.arms).sort(),['C','N','R']);
 for(const arm of ['C','N','R'])await validateSaved(s,arm,seal.arms[arm],g);
 repeatEqual(seal.arms.N,seal.arms.R);
 if(m.canaries.includes(s.name)){
  assert.notEqual(seal.arms.N.native_sha256,s.G.native_sha256,'Canary neutral arm has no native pixel effect');
  assert.notEqual(seal.arms.N.working_payload_sha256,s.G.working_payload_sha256,'Canary neutral arm has no working pixel effect');
 }
}
async function reuseSeal(s,folder,g){
 if(s.kind==='G')return {manifest_sha256:expected,name:s.name,role:'fit',kind:'G',source_raf_sha256:s.raf_sha256,identity:s.G};
 const old=s.reuse.source_record,arms={};
 for(const arm of ['C','N','R']){
  const z=s.kind==='W'?old.arms[{C:'control',N:'W',R:'W-repeat'}[arm]]:old.arms[{C:'C',N:'P',R:'R'}[arm]];
  const r={file:z.file,sha256:z.sha256,native_sha256:z.native_sha256,working_payload_sha256:z.working_payload_sha256,
   native_size:z.native_size,width:z.width,height:z.height,options_json:z.options_json,options_sha256:z.options_sha256,
   strips_file:s.kind==='W'?z.strips.file:z.strips_file,strips_sha256:s.kind==='W'?z.strips.sha256:z.strips_sha256,
   strip_rows:s.kind==='W'?z.strips.rows:z.strip_rows,source_coefficients:{autoRgbg:s.kind==='W'?z.source_coefficients.autoRgbg:z.source_coefficients.autoRgbg,asShotCamMul:z.source_coefficients.asShotCamMul},
   returned_metadata:s.kind==='W'?null:z.returned_metadata,metadata_note:s.kind==='W'?'Not returned by old W helper; camera/WB/cam validated internally, not reconstructed.':'Actual stored decoder metadata; options are declared, not applied-parameter readback.',contract:arm==='C'?null:s.contract};
  if(arm!=='C'){
   r.sidecar_sha256=await hashFile(safeArtifact(r.file)+'.json');
   if(s.kind==='W')assert.deepEqual(await json(safeArtifact(r.file)+'.json'),z.provenance);
  }
  await validateSaved(s,arm,r,g);arms[arm]=r;
 }
 repeatEqual(arms.N,arms.R);
 const seal={manifest_sha256:expected,name:s.name,role:'fit',kind:s.kind,source_raf_sha256:s.raf_sha256,reused_qualified_artifact:true,arms};
 await validateSeal(s,seal,g);return seal;
}
const modules={'/decode-w.mjs':'scripts/lib/decode-provia-v2-research.mjs','/decode-p.mjs':'scripts/lib/decode-fjnrgb-profile-research.mjs',
 '/codec.mjs':'scripts/lib/fjlrg-v2.mjs','/fjlrg-v2.mjs':'scripts/lib/fjlrg-v2.mjs','/profile-codec.mjs':'scripts/lib/fjnrgb-profile.mjs','/wb.mjs':'scripts/lib/raf-wb-coefficients.mjs'};
let activeSource=null;
const server=createServer(async(req,res)=>{try{
 assert.equal(req.method,'GET');const u=new URL(req.url,'http://127.0.0.1');
 if(u.pathname==='/'){res.writeHead(200,{'Content-Type':'text/html'});res.end('<!doctype html><link rel="icon" href="data:,">');return;}
 let p;if(modules[u.pathname])p=resolve(root,modules[u.pathname]);
 else if(u.pathname.startsWith('/raw/')){const name=decodeURIComponent(u.pathname.slice(5));assert.equal(name,activeSource,'RAF outside active scene');p=sources.get(name);}
 else if(u.pathname.startsWith('/libraw/')){const n=u.pathname.slice(8);assert.equal(n,basename(n));assert(['index.js','worker.js','libraw.js','libraw.wasm'].includes(n));p=resolve(root,'node_modules/libraw-wasm/dist',n);}
 assert(p);const st=await stat(p);res.writeHead(200,{'Content-Length':st.size,'Content-Type':['.mjs','.js'].includes(extname(p))?'text/javascript':extname(p)==='.wasm'?'application/wasm':'application/octet-stream'});createReadStream(p).pipe(res);
 }catch{if(!res.headersSent){res.writeHead(404);res.end();}}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin=`http://127.0.0.1:${server.address().port}`;
function bounded(p,ms,label){let t;return Promise.race([p,new Promise((_,no)=>{t=setTimeout(()=>no(Error(label+' timeout')),ms);})]).finally(()=>clearTimeout(t));}
async function decode(s,arm){
 let bs,b,page,result;const options=armOptions(s,arm),scene={...s,g_native_size:s.expected_native_size,capture_dynamic_range_source:'frozen approved train manifest captureDynamicRange'};
 assert.equal(activeSource,null);activeSource=s.name;
 try{
  const launching=chromium.launchServer({headless:true,host:'127.0.0.1'});
  try{bs=await bounded(launching,60000,'launch');}catch(e){launching.then(x=>x.kill()).catch(()=>{});throw e;}
  b=await bounded(chromium.connect(bs.wsEndpoint()),60000,'connect');page=await bounded(b.newPage(),60000,'new page');await bounded(page.goto(origin),60000,'goto');
  const r=await bounded(page.evaluate(async({scene,arm,options,wasmSha})=>{
   if(scene.kind==='W')return (await import('/decode-w.mjs')).decodeResearchRaw({scene,options,wasmSha,inputContract:'v2-linear-autowb',control:arm==='C'});
   return (await import('/decode-p.mjs')).decodeFjnrgbProfileResearch({scene,arm:arm==='C'?'control':'treatment',wasmSha256:wasmSha});
  },{scene,arm,options,wasmSha:m.wasm_sha256}),600000,'decode');
  result=await normalizeDecoderResult(s,arm,r,{wasmSha256:m.wasm_sha256});
 }finally{
  if(page)await bounded(page.close(),10000,'page close').catch(()=>{});
  if(b)await bounded(b.close(),10000,'browser close').catch(()=>{});
  if(bs){await bounded(bs.close(),10000,'server close').catch(()=>{});if(bs.process().exitCode===null&&bs.process().signalCode===null)await bounded(bs.kill(),10000,'owned browser kill');assert(bs.process().exitCode!==null||bs.process().signalCode!==null,'Owned browser still running');}
  activeSource=null;
 }
 return result;
}
const seals={},progress={manifest_sha256:expected,completed:false,scene_seal_sha256:seals,canary_only:canaryOnly};
const saveProgress=()=>writeFile(resolve(out,'progress.json'),JSON.stringify({...progress,updated_utc:new Date().toISOString()},null,2)+'\n');
try{
 // Reuse is verified before any new costly decode; never access old development artifacts.
 const ordered=[...m.scenes.filter(s=>!s.requires_new_decode),...m.scenes.filter(s=>s.requires_new_decode&&(!canaryOnly||m.canaries.includes(s.name)))];
 for(const s of ordered){
  const file=sources.get(s.name),folder=resolve(out,s.shoot),sealPath=resolve(folder,'scene-seal.json');
  assert.equal(await hashFile(file),s.raf_sha256);const g=await originalG(s);await mkdir(folder,{recursive:true});
  if(await exists(sealPath))await validateSeal(s,await json(sealPath),g);
  else if(!s.requires_new_decode){await writeOnce(sealPath,await reuseSeal(s,folder,g));}
  else{
   const arms={};
   for(const arm of ['C','N','R']){
    const armSeal=resolve(folder,arm+'-arm.json');
    if(await exists(armSeal)){
     const saved=await json(armSeal);assert.equal(saved.manifest_sha256,expected);assert.equal(saved.name,s.name);assert.equal(saved.arm,arm);
     await validateSaved(s,arm,saved.artifact,g);arms[arm]=saved.artifact;
    }else{
     const base=resolve(folder,arm+(arm==='C'?'-control.u16le':s.kind==='W'?'-neutral.fjlrg':'-neutral.fjnrgb'));
     const strip=resolve(folder,arm+'-native-strips.u16le');
     assert(!await exists(base)&&!await exists(base+'.json')&&!await exists(strip),'Unsealed prior output exists; investigate, never overwrite');
     const started=Date.now(),r=await decode(s,arm),{bytes,strips,provenance,...meta}=r;
     assert(Buffer.isBuffer(bytes)&&Buffer.isBuffer(strips));
     const artifact={...meta,file:relative(root,base),sha256:sha(bytes),strips_file:relative(root,strip),strips_sha256:sha(strips)};
     await writeFile(base,bytes,{flag:'wx'});await writeFile(strip,strips,{flag:'wx'});
     if(arm!=='C'){assert(provenance);artifact.sidecar_sha256=await writeOnce(base+'.json',provenance);}
     await validateSaved(s,arm,artifact,g);if(arm==='R')repeatEqual(arms.N,artifact);
     await writeOnce(armSeal,{manifest_sha256:expected,name:s.name,arm,artifact,seconds:(Date.now()-started)/1000});arms[arm]=artifact;
    }
    console.log('NEUTRAL_WB_ARM',s.name,s.kind,arm,'verified');
   }
   const seal={manifest_sha256:expected,name:s.name,role:'fit',kind:s.kind,source_raf_sha256:s.raf_sha256,arms};await validateSeal(s,seal,g);await writeOnce(sealPath,seal);
  }
  assert.equal(await hashFile(file),s.raf_sha256);seals[s.name]=await hashFile(sealPath);await saveProgress();console.log('NEUTRAL_WB_SCENE',Object.keys(seals).length,ordered.length,s.name,s.kind,'sealed');
 }
 if(canaryOnly){assert.equal(Object.keys(seals).length,16);console.log('NEUTRAL_WB_CANARIES_CAPTURED; independent verifier required before bulk');}
 else{
  assert.equal(Object.keys(seals).length,455);progress.completed=true;const complete=resolve(out,'complete.json');
  if(await exists(complete))assert.deepEqual(await json(complete),progress);else await writeOnce(complete,progress);
  await saveProgress();console.log('NEUTRAL_WB_FIT_CAPTURE_COMPLETE',455);
 }
}catch(e){progress.stopped={reason:String(e),stack:e.stack};await saveProgress();throw e;}
finally{server.closeAllConnections?.();await bounded(new Promise(r=>server.close(r)),10000,'HTTP close');}
