#!/usr/bin/env node
/** Gamma-only G input exports, fixed583 train sources, no target/model access. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream,existsSync} from 'node:fs';
import {readFile,writeFile,mkdir,stat,rename} from 'node:fs/promises';
import {resolve,basename,extname,relative} from 'node:path';
import {createServer} from 'node:http';
import {execFileSync} from 'node:child_process';
import {chromium} from 'playwright';
import {decodeFjlrgV2} from './lib/fjlrg-v2.mjs';
const root=process.cwd(),manifestPath='training/diagnostics/provia-v2-train-export-manifest.json';
const hash=b=>createHash('sha256').update(b).digest('hex');
const fileHash=async p=>{const h=createHash('sha256');for await(const b of createReadStream(p))h.update(b);return h.digest('hex');};
const expected=process.argv[2],resume=process.argv[3]==='--resume';assert(process.argv.length===3||(process.argv.length===4&&resume));
assert.equal(await fileHash(manifestPath),expected);const manifest=JSON.parse(await readFile(manifestPath,'utf8'));
for(const[p,h]of Object.entries(manifest.bindings))assert.equal(await fileHash(p),h,`Bound file drift ${p}`);
for(const p of [manifestPath,'scripts/export-provia-input-v2-train.mjs','training/verify_provia_v2_train_exports.py'])assert.equal(hash(execFileSync('git',['show',`HEAD:${p}`])),await fileHash(p),'Uncommitted frozen code');
assert.equal(manifest.scenes.length,583);assert.deepEqual(manifest.scenes.slice(0,3).map(s=>s.name),manifest.canaries);
const allowed=new Map(manifest.scenes.map(s=>[s.name,resolve(root,'calibration-input',s.shoot,s.name)]));assert.equal(allowed.size,583);
const output=resolve(root,manifest.output_dir);assert.equal(output,resolve(root,'training/checkpoints/provia-input-v2-train-g'));
if(resume){assert.equal(JSON.parse(await readFile(resolve(output,'run-anchor.json'),'utf8')).manifest_sha256,expected);assert(!existsSync(resolve(output,'complete.json')));}
else{await mkdir(output);await writeFile(resolve(output,'run-anchor.json'),JSON.stringify({manifest_sha256:expected})+'\n',{flag:'wx'});}
const server=createServer(async(req,res)=>{
 try{
  const u=new URL(req.url,'http://localhost');let p;
  if(u.pathname==='/'){res.writeHead(200,{'Content-Type':'text/html'});res.end('<!doctype html><link rel="icon" href="data:,">');return;}
  const modules={'/decode.mjs':'scripts/lib/decode-provia-v2-research.mjs','/codec.mjs':'scripts/lib/fjlrg-v2.mjs','/wb.mjs':'scripts/lib/raf-wb-coefficients.mjs'};
  if(modules[u.pathname])p=resolve(root,modules[u.pathname]);
  else if(u.pathname.startsWith('/raw/'))p=allowed.get(decodeURIComponent(u.pathname.slice(5)));
  else if(u.pathname.startsWith('/libraw/')){const n=decodeURIComponent(u.pathname.slice(8));assert.equal(n,basename(n));assert(['index.js','worker.js','libraw.js','libraw.wasm'].includes(n));p=resolve(root,'node_modules/libraw-wasm/dist',n);}
  assert(p);const s=await stat(p);res.writeHead(200,{'Content-Length':s.size,'Content-Type':['.js','.mjs'].includes(extname(p))?'text/javascript':extname(p)==='.wasm'?'application/wasm':'application/octet-stream'});createReadStream(p).pipe(res);
 }catch{res.writeHead(404);res.end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const browser=await chromium.launch({headless:true});
const report={manifest_sha256:expected,completed:false,scope:manifest.scope,scenes:[]};
async function validate(scene,row){
 assert.equal(row.name,scene.name);assert.equal(row.shoot,scene.shoot);
 const f=resolve(root,row.file),strip=resolve(root,row.strips.file);assert.equal(resolve(f,'..'),resolve(output,scene.shoot));assert.equal(resolve(strip,'..'),resolve(output,scene.shoot));
 assert.equal(await fileHash(f),row.sha256);assert.equal(await fileHash(strip),row.strips.sha256);
 const provenance=JSON.parse(await readFile(f+'.json','utf8'));
 const frame=await decodeFjlrgV2(new Uint8Array(await readFile(f)),provenance,{expectedContract:'v2-linear-asshot',expectedSourceSha256:scene.raf_sha256,expectedWasmSha256:manifest.wasm_sha256});
 assert.deepEqual([frame.width,frame.height],scene.fjlrg_wh);assert.deepEqual([row.width,row.height],scene.fjlrg_wh);
 assert.deepEqual(provenance.wb_source.levels_rgbg,scene.cam_mul);assert.deepEqual(row.source_coefficients.asShotCamMul,scene.cam_mul);assert.deepEqual(row.source_coefficients.autoRgbg,scene.auto_rgbg);
 assert.equal(provenance.source_wb_mode,scene.source_wb_mode);assert.deepEqual(provenance.source_wb_fine_tune_raw,scene.source_wb_fine_tune_raw);assert.equal(provenance.source_kelvin,scene.source_kelvin);
 assert.deepEqual(provenance.declared_options,manifest.base_options);assert.deepEqual(provenance.decoded_native_size,row.native_size);assert.equal(provenance.capture_dynamic_range,scene.captureDynamicRange);assert.equal(scene.capture_dynamic_range,scene.captureDynamicRange);
 if(scene.expected_pilot_g_sha256)assert.equal(row.sha256,scene.expected_pilot_g_sha256,'Independent P1b helper canary drift');
 assert.equal(await fileHash(allowed.get(scene.name)),scene.raf_sha256,'Source changed after decode');
}
try{
 for(const scene of manifest.scenes){
  assert.equal(await fileHash(allowed.get(scene.name)),scene.raf_sha256,'Source changed before decode');
  const folder=resolve(output,scene.shoot),seal=resolve(folder,'scene-complete.json');
  if(resume&&existsSync(seal)&&existsSync(seal+'.sha256')){
   const b=await readFile(seal);assert.equal(hash(b),(await readFile(seal+'.sha256','utf8')).trim());const row=JSON.parse(b);await validate(scene,row);report.scenes.push(row);console.log(`Resume verified ${report.scenes.length}/583 ${scene.name}`);continue;
  }
  if(existsSync(folder)){assert(resume);const partials=resolve(output,'partials');await mkdir(partials,{recursive:true});await rename(folder,resolve(partials,`${scene.shoot}.${Date.now()}`));}
  await mkdir(folder);let row={name:scene.name,shoot:scene.shoot};const page=await browser.newPage(),started=Date.now();
  try{
   await page.goto(`http://127.0.0.1:${server.address().port}`);
   const r=await page.evaluate(async a=>(await import('/decode.mjs')).decodeResearchRaw(a),{scene,options:manifest.base_options,wasmSha:manifest.wasm_sha256,inputContract:'v2-linear-asshot'});
   const b=Buffer.from(r.base64,'base64'),strips=Buffer.from(r.stripsBase64,'base64');
   const file=resolve(folder,'input-v2-linear-asshot.fjlrg'),stripFile=resolve(folder,'G-native-strips.u16le');
   await writeFile(file,b,{flag:'wx'});await writeFile(file+'.json',JSON.stringify(r.provenance,null,2)+'\n',{flag:'wx'});await writeFile(stripFile,strips,{flag:'wx'});
   const{base64,stripsBase64,stripRows,...info}=r;
   row={...row,...info,file:relative(root,file),sha256:hash(b),strips:{file:relative(root,stripFile),sha256:hash(strips),rows:stripRows,width:r.native_size[0],channels:3,dtype:'uint16-le'},seconds:(Date.now()-started)/1000};
   await writeFile(resolve(folder,'scene-measured-unsealed.json'),JSON.stringify(row,null,2)+'\n',{flag:'wx'});
   await validate(scene,row);row.passed=true;
   const rb=Buffer.from(JSON.stringify(row,null,2)+'\n');await writeFile(seal,rb,{flag:'wx'});await writeFile(seal+'.sha256',hash(rb)+'\n',{flag:'wx'});
   report.scenes.push(row);await writeFile(resolve(output,'progress.json'),JSON.stringify(report,null,2)+'\n');
   console.log(`${report.scenes.length}/583 ${scene.name} ${row.seconds.toFixed(1)}s${scene.expected_pilot_g_sha256?' exact P1b canary':''}`);
  }catch(e){await writeFile(resolve(folder,'failure.json'),JSON.stringify({name:scene.name,error:String(e),measurements:row},null,2)+'\n',{flag:'wx'});throw e;}
  finally{await page.close();}
 }
 report.completed=true;await writeFile(resolve(output,'complete.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
}finally{await browser.close();await new Promise(r=>server.close(r));}
