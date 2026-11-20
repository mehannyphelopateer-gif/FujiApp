#!/usr/bin/env node
/** F11 source-Auto W input pilot. No target pixels, model or live app changes. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream,existsSync} from 'node:fs';
import {readFile,writeFile,mkdir,stat,rename} from 'node:fs/promises';
import {resolve,basename,extname,relative} from 'node:path';
import {createServer} from 'node:http';
import {execFileSync} from 'node:child_process';
import {chromium} from 'playwright';
import {decodeFjlrgV2} from './lib/fjlrg-v2.mjs';
const root=process.cwd(),manifestPath='training/diagnostics/provia-v2-wb-pilot-manifest.json';
const hash=b=>createHash('sha256').update(b).digest('hex');
const fileHash=async p=>{const h=createHash('sha256');for await(const b of createReadStream(p))h.update(b);return h.digest('hex');};
const expected=process.argv[2],resume=process.argv[3]==='--resume';
assert(process.argv.length===3||(process.argv.length===4&&resume));
assert.equal(await fileHash(manifestPath),expected);
const manifest=JSON.parse(await readFile(manifestPath,'utf8'));
for(const[p,h]of Object.entries(manifest.bindings))assert.equal(await fileHash(p),h,`Bound file drift ${p}`);
for(const p of [manifestPath,'scripts/export-provia-input-v2-wb-pilot.mjs','training/verify_provia_v2_wb_pilot.py'])assert.equal(hash(execFileSync('git',['show',`HEAD:${p}`])),await fileHash(p),'Uncommitted frozen code');
const output=resolve(root,manifest.output_dir);assert.equal(output,resolve(root,'training/checkpoints/provia-input-v2-wb-pilot'));
assert.equal(manifest.scenes.length,11);assert(manifest.scenes.every(s=>s.source_wb_mode===0));
const allowed=new Map(manifest.scenes.map(s=>[s.name,resolve(root,'calibration-input',s.shoot,s.name)]));assert.equal(allowed.size,11);
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
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true});
const report={manifest_sha256:expected,completed:false,scope:manifest.scope,scenes:[]};
async function validateRow(scene,row){
 assert.equal(row.name,scene.name);assert.equal(row.shoot,scene.shoot);assert.deepEqual(Object.keys(row.arms).sort(),['control','W','W-repeat'].sort());
 const baseline=resolve(root,scene.g_file);assert.equal(await fileHash(baseline),scene.g_sha256);
 const old=new Uint8Array(await readFile(baseline));
 const g=await decodeFjlrgV2(old,JSON.parse(await readFile(baseline+'.json','utf8')),{expectedContract:'v2-linear-asshot',expectedSourceSha256:scene.raf_sha256,expectedWasmSha256:manifest.wasm_sha256});
 const control=row.arms.control;
 assert.equal(control.sha256,hash(old.subarray(64)),'Explicit as-shot control working payload differs from G');
 assert.equal(control.strips.sha256,scene.g_strips_sha256,'Explicit as-shot control native strips differ from G');
 const dims=[g.width,g.height];
 for(const[arm,r]of Object.entries(row.arms)){
  assert.deepEqual([r.width,r.height],dims);assert.deepEqual(r.native_size,scene.g_native_size);
  const f=resolve(root,r.file);assert.equal(resolve(f,'..'),resolve(output,scene.shoot));assert.equal(await fileHash(f),r.sha256);
  const strip=resolve(root,r.strips.file);assert.equal(resolve(strip,'..'),resolve(output,scene.shoot));assert.equal(await fileHash(strip),r.strips.sha256);
  if(arm!=='control')await decodeFjlrgV2(new Uint8Array(await readFile(f)),JSON.parse(await readFile(f+'.json','utf8')),{expectedContract:'v2-linear-autowb',expectedSourceSha256:scene.raf_sha256,expectedWasmSha256:manifest.wasm_sha256});
 }
 assert.equal(row.arms.W.sha256,row.arms['W-repeat'].sha256,'W repeat containers differ');
 assert.equal(row.arms.W.native_sha256,row.arms['W-repeat'].native_sha256,'W native repeat differs');
 assert.equal(row.arms.W.strips.sha256,row.arms['W-repeat'].strips.sha256,'W strip repeat differs');
 assert.equal(await fileHash(allowed.get(scene.name)),scene.raf_sha256);
}
try{
 for(const scene of manifest.scenes){
  assert.equal(await fileHash(allowed.get(scene.name)),scene.raf_sha256);
  const folder=resolve(output,scene.shoot),sealed=resolve(folder,'scene-complete.json');
  if(resume&&existsSync(sealed)&&existsSync(sealed+'.sha256')){
   const b=await readFile(sealed);assert.equal(hash(b),(await readFile(sealed+'.sha256','utf8')).trim());const row=JSON.parse(b);await validateRow(scene,row);report.scenes.push(row);console.log(`Resume verified ${scene.name}`);continue;
  }
  if(existsSync(folder)){assert(resume);const partials=resolve(output,'partials');await mkdir(partials,{recursive:true});await rename(folder,resolve(partials,`${scene.shoot}.${Date.now()}`));}
  await mkdir(folder);const row={name:scene.name,shoot:scene.shoot,arms:{}};
  try{
   for(const arm of ['control','W','W-repeat']){
    const isControl=arm==='control';
    const options={...manifest.base_options,useCameraWb:false,userMul:isControl?[...scene.cam_mul.slice(0,3),scene.cam_mul[1]]:scene.auto_rgbg};
    const page=await browser.newPage(),started=Date.now();
    try{
     await page.goto(`http://127.0.0.1:${server.address().port}`);
     const r=await page.evaluate(async args=>{const{decodeResearchRaw}=await import('/decode.mjs');return decodeResearchRaw(args);},{scene,options,wasmSha:manifest.wasm_sha256,inputContract:'v2-linear-autowb',control:isControl});
     const b=Buffer.from(r.base64,'base64'),strips=Buffer.from(r.stripsBase64,'base64');
     const file=resolve(folder,isControl?'explicit-asshot-control.u16le':arm==='W'?'input-v2-linear-autowb.fjlrg':'input-v2-linear-autowb-repeat.fjlrg');
     await writeFile(file,b,{flag:'wx'});if(r.provenance)await writeFile(file+'.json',JSON.stringify(r.provenance,null,2)+'\n',{flag:'wx'});
     const stripFile=resolve(folder,`${arm}-native-strips.u16le`);await writeFile(stripFile,strips,{flag:'wx'});
     const{base64,stripsBase64,stripRows,...info}=r;
     row.arms[arm]={...info,file:relative(root,file),sha256:hash(b),strips:{file:relative(root,stripFile),sha256:hash(strips),rows:stripRows,width:r.native_size[0],channels:3,dtype:'uint16-le'},seconds:(Date.now()-started)/1000};
     console.log(`${report.scenes.length+1}/11 ${scene.name} ${arm} ${row.arms[arm].seconds.toFixed(1)}s`);
    }finally{await page.close();}
   }
   await writeFile(resolve(folder,'scene-measured-unsealed.json'),JSON.stringify(row,null,2)+'\n',{flag:'wx'});
   await validateRow(scene,row);row.gates_passed=true;
   const b=Buffer.from(JSON.stringify(row,null,2)+'\n');await writeFile(sealed,b,{flag:'wx'});await writeFile(sealed+'.sha256',hash(b)+'\n',{flag:'wx'});
   report.scenes.push(row);await writeFile(resolve(output,'progress.json'),JSON.stringify(report,null,2)+'\n');
  }catch(e){await writeFile(resolve(folder,'failure.json'),JSON.stringify({name:scene.name,error:String(e),measurements:row},null,2)+'\n',{flag:'wx'});throw e;}
 }
 report.completed=true;await writeFile(resolve(output,'complete.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
}finally{await browser.close();await new Promise(r=>server.close(r));}
