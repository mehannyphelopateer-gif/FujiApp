#!/usr/bin/env node
/** Frozen four-FIT-source RGB identity/repeat qualification. No targets/models. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {createServer} from 'node:http';
import {createReadStream} from 'node:fs';
import {readFile,writeFile,mkdir,stat} from 'node:fs/promises';
import {resolve,relative,basename,extname} from 'node:path';
import {chromium} from 'playwright';
import * as V2 from './lib/fjlrg-v2.mjs';
import * as P from './lib/fjnrgb-profile.mjs';
import {armOptions,validateSourceScene,stripRows} from './lib/decode-fjnrgb-profile-research.mjs';
const root=process.cwd(),rel='training/diagnostics/profile-rgb-sentinel-manifest.json';
const sha=b=>createHash('sha256').update(b).digest('hex');
const hashFile=async p=>{const h=createHash('sha256');for await(const b of createReadStream(p))h.update(b);return h.digest('hex');};
const expected=process.argv[2];assert(process.argv.length===3&&/^[0-9a-f]{64}$/.test(expected));
assert.equal(await hashFile(rel),expected);const m=JSON.parse(await readFile(rel,'utf8'));
for(const[p,h]of Object.entries(m.bindings))assert.equal(await hashFile(p),h,`Bound drift ${p}`);
for(const p of [rel,'scripts/capture-profile-rgb-sentinels.mjs','training/qualify_profile_rgb_sentinels.py'])assert.equal(sha(execFileSync('git',['show',`HEAD:${p}`])),await hashFile(p),`Uncommitted ${p}`);
assert.equal(m.scenes.length,4);assert.equal(new Set(m.scenes.map(s=>s.session)).size,4);
const out=resolve(root,m.output_dir);assert.equal(out,resolve(root,'training/checkpoints/profile-rgb-sentinels'));
const sources=new Map(),args=new Map(),oldPayload=new Map();
for(const s of m.scenes){
 assert.match(s.shoot,/^Shoot \d+$/);assert.match(s.name,/^DSCF\d{4}\.RAF$/);
 const a=Object.fromEntries(['name','role','raf_sha256','cam_mul','auto_rgbg','source_wb_mode','source_kelvin','source_wb_fine_tune_raw','capture_dynamic_range'].map(k=>[k,s[k]]));
 a.g_native_size=s.expected_native_size;a.capture_dynamic_range_source='frozen approved train manifest captureDynamicRange';
 validateSourceScene(a);args.set(s.name,a);sources.set(s.name,resolve(root,'calibration-input',s.shoot,s.name));
 // Original G validation is restricted to the exact four allowlisted fit inputs.
 assert.equal(await hashFile(s.G.file),s.G.sha256);assert.equal(await hashFile(s.G.file+'.json'),s.G.sidecar_sha256);
 const frame=await V2.decodeFjlrgV2(new Uint8Array(await readFile(s.G.file)),JSON.parse(await readFile(s.G.file+'.json','utf8')),
  {expectedContract:'v2-linear-asshot',expectedSourceSha256:s.raf_sha256,expectedWasmSha256:m.wasm_sha256});
 assert.deepEqual([frame.width,frame.height],s.expected_working_size);
 const b=Buffer.from(frame.data.buffer,frame.data.byteOffset,frame.data.byteLength);assert.equal(sha(b),s.G.working_payload_sha256);oldPayload.set(s.name,Buffer.from(b));
 assert.equal(await hashFile(s.G.strips.file),s.G.strips.sha256);
}
await mkdir(out); // refuses any previous output: never overwrite research evidence
const modules={'/decode.mjs':'scripts/lib/decode-fjnrgb-profile-research.mjs','/profile-codec.mjs':'scripts/lib/fjnrgb-profile.mjs',
 '/fjlrg-v2.mjs':'scripts/lib/fjlrg-v2.mjs','/wb.mjs':'scripts/lib/raf-wb-coefficients.mjs'};
const server=createServer(async(req,res)=>{try{
 assert.equal(req.method,'GET');const u=new URL(req.url,'http://127.0.0.1');
 if(u.pathname==='/'){res.writeHead(200,{'Content-Type':'text/html'});res.end('<!doctype html><link rel="icon" href="data:,">');return;}
 let p;if(modules[u.pathname])p=resolve(root,modules[u.pathname]);
 else if(u.pathname.startsWith('/raw/'))p=sources.get(decodeURIComponent(u.pathname.slice(5)));
 else if(u.pathname.startsWith('/libraw/')){const n=decodeURIComponent(u.pathname.slice(8));assert.equal(n,basename(n));assert(['index.js','worker.js','libraw.js','libraw.wasm'].includes(n));p=resolve(root,'node_modules/libraw-wasm/dist',n);}
 assert(p);const st=await stat(p);res.writeHead(200,{'Content-Length':st.size,'Content-Type':['.js','.mjs'].includes(extname(p))?'text/javascript':extname(p)==='.wasm'?'application/wasm':'application/octet-stream'});createReadStream(p).pipe(res);
 }catch{if(!res.headersSent){res.writeHead(404);res.end();}}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin=`http://127.0.0.1:${server.address().port}`;
function timeout(p,ms,label){let timer;return Promise.race([p,new Promise((_,no)=>{timer=setTimeout(()=>no(new Error(`${label} timeout`)),ms);})]).finally(()=>clearTimeout(timer));}
async function boundedClose(label,fn,warnings){try{await timeout(Promise.resolve().then(fn),10000,label);return true;}catch(e){warnings.push(String(e));return false;}}
async function ownPage(fn){
 const warnings=[];let bs,b,page,error,result;
 const launch=chromium.launchServer({headless:true,host:'127.0.0.1'});
 try{try{bs=await timeout(launch,60000,'launch');}catch(e){launch.then(s=>s.kill()).catch(()=>{});throw e;}
 b=await timeout(chromium.connect(bs.wsEndpoint()),60000,'connect');page=await timeout(b.newPage(),60000,'newPage');
 await timeout(page.goto(origin),60000,'goto');result=await timeout(fn(page),600000,'decode/return');
 }catch(e){error=e;}
 finally{
  if(page)await boundedClose('page.close',()=>page.close(),warnings);
  if(b)await boundedClose('browser.close',()=>b.close(),warnings);
  if(bs){const proc=bs.process();const closed=await boundedClose('own server.close',()=>bs.close(),warnings);
   if(!closed||proc.exitCode===null&&proc.signalCode===null)await boundedClose('own server.kill',()=>bs.kill(),warnings);
   if(proc.exitCode===null&&proc.signalCode===null)error??=new Error('Own browser exit unconfirmed');}
 }
 if(error)throw error;return {result,warnings};
}
const report={manifest_sha256:expected,completed:false,scenes:[],scope:m.scope};
const progress=()=>writeFile(resolve(out,'progress.json'),JSON.stringify(report,null,2)+'\n');
try{
 for(const s of m.scenes){
  assert.equal(await hashFile(sources.get(s.name)),s.raf_sha256);const folder=resolve(out,s.shoot);await mkdir(folder);
  const row={name:s.name,arms:{}};report.scenes.push(row);
  for(const arm of ['C','P','R']){
   const kind=arm==='C'?'control':'treatment';
   const {result:r,warnings}=await ownPage(page=>page.evaluate(async a=>(await import('/decode.mjs')).decodeFjnrgbProfileResearch(a),
    {scene:args.get(s.name),arm:kind,wasmSha256:m.wasm_sha256}));
   assert.equal(r.arm,kind);assert.equal(r.scene_name,s.name);assert.equal(r.source_raf_sha256,s.raf_sha256);assert.equal(r.parameters_are_declared,true);
   assert.match(r.native_sha256,/^[0-9a-f]{64}$/);assert.match(r.working_payload_sha256,/^[0-9a-f]{64}$/);
   assert.deepEqual(r.declared_options,armOptions(args.get(s.name),kind));assert.equal(r.options_json,V2.serializeOptions(r.declared_options));assert.equal(sha(Buffer.from(r.options_json)),r.options_sha256);
   assert.deepEqual(r.native_size,s.expected_native_size);assert.deepEqual(r.working_size,s.expected_working_size);
   assert.deepEqual(r.decoder_readback.cam_mul,s.cam_mul);assert.equal(r.decoder_readback.camera_model,'X100VI');assert.equal(r.decoder_readback.fujiWB,4080);
   assert.deepEqual(r.source_coefficients.as_shot_cam_mul,s.cam_mul);assert.deepEqual(r.source_coefficients.auto_rgbg,s.auto_rgbg);
   assert.deepEqual(r.strips.rows,stripRows(r.native_size[1]));assert.equal(r.strips.width,r.native_size[0]);assert.equal(r.strips.channels,3);
   const strips=Buffer.from(r.strips.base64,'base64');assert.equal(strips.length,r.strips.rows.length*r.native_size[0]*6);
   const strip=resolve(folder,`${arm}-native-strips.u16le`);await writeFile(strip,strips,{flag:'wx'});
   let bytes,file;
   if(arm==='C'){
    assert(!r.container);bytes=Buffer.from(r.working_payload.base64,'base64');file=resolve(folder,'C-control.u16le');
    assert.deepEqual(bytes,oldPayload.get(s.name));assert.equal(r.native_sha256,s.G.native_sha256);assert.equal(sha(strips),s.G.strips.sha256);
   }else{
    assert.equal(r.container?.contract,P.CONTRACT);
    assert(!r.working_payload);bytes=Buffer.from(r.container.base64,'base64');file=resolve(folder,`${arm}-profile.fjnrgb`);
    const frame=await P.decodeFjnrgbProfile(new Uint8Array(bytes),r.container.provenance,{expectedSourceSha256:s.raf_sha256,expectedWasmSha256:m.wasm_sha256});
    assert.equal(r.container.container_sha256,sha(bytes));assert.equal(r.container.provenance.options_json,r.options_json);
    assert.equal(sha(Buffer.from(frame.data.buffer,frame.data.byteOffset,frame.data.byteLength)),r.working_payload_sha256);
    await writeFile(file+'.json',JSON.stringify(r.container.provenance,null,2)+'\n',{flag:'wx'});
   }
   await writeFile(file,bytes,{flag:'wx'});assert.equal(await hashFile(file),sha(bytes));assert.equal(await hashFile(strip),sha(strips));
   const measured={file:relative(root,file),sha256:sha(bytes),strips_file:relative(root,strip),strips_sha256:sha(strips),strip_rows:r.strips.rows,
    native_size:r.native_size,width:r.working_size[0],height:r.working_size[1],native_sha256:r.native_sha256,working_payload_sha256:r.working_payload_sha256,
    declared_options:r.declared_options,options_json:r.options_json,options_sha256:r.options_sha256,returned_metadata:r.decoder_readback,
    source_coefficients:{asShotCamMul:r.source_coefficients.as_shot_cam_mul,autoRgbg:r.source_coefficients.auto_rgbg},cleanup_warnings:warnings,note:r.note};
   if(arm==='C')assert.equal(sha(bytes),r.working_payload_sha256);
   if(arm==='R'){for(const key of ['sha256','strips_sha256','native_sha256','working_payload_sha256'])assert.equal(measured[key],row.arms.P[key],`Repeat ${key}`);}
   if(arm!=='C'){assert.notEqual(r.native_sha256,s.G.native_sha256);assert.notEqual(r.working_payload_sha256,s.G.working_payload_sha256);}
   row.arms[arm]=measured;await progress();console.log(`RGB_SENTINEL ${s.name} ${arm} verified`);
  }
  row.raf_sha256_after=await hashFile(sources.get(s.name));assert.equal(row.raf_sha256_after,s.raf_sha256);await progress();
 }
 report.completed=true;await writeFile(resolve(out,'complete.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log('RGB_SENTINEL_CAPTURE_COMPLETE');
}catch(e){report.stopped={reason:String(e)};await progress();throw e;}
finally{server.closeAllConnections?.();await timeout(new Promise(r=>server.close(r)),10000,'HTTP server.close');}
