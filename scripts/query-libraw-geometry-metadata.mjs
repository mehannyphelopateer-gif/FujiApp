#!/usr/bin/env node
/** Frozen fit-only metadata query. Never calls any pixel extraction method. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {createServer} from 'node:http';
import {createReadStream} from 'node:fs';
import {readFile,writeFile,stat} from 'node:fs/promises';
import {resolve,basename,extname} from 'node:path';
import {chromium} from 'playwright';
const root=process.cwd(),manifest='training/diagnostics/decoder-geometry-metadata-manifest.json';
const hash=async p=>{const h=createHash('sha256');for await(const b of createReadStream(p))h.update(b);return h.digest('hex');};
const sha=b=>createHash('sha256').update(b).digest('hex');
const expected=process.argv[2];assert(process.argv.length===3&&/^[0-9a-f]{64}$/.test(expected));
assert.equal(await hash(manifest),expected);const m=JSON.parse(await readFile(manifest,'utf8'));
assert.equal(sha(execFileSync('git',['show',`HEAD:${manifest}`])),expected);
for(const [p,h]of Object.entries(m.bindings)){
 assert.equal(await hash(p),h,`Binding drift ${p}`);
 if(!p.startsWith('node_modules/'))assert.equal(sha(execFileSync('git',['show',`HEAD:${p}`])),h,`Uncommitted ${p}`);
}
assert.equal(m.scenes.length,8);assert.equal(new Set(m.scenes.map(s=>s.name)).size,8);
const sources=new Map();
for(const s of m.scenes){assert.equal(s.role,'fit');assert.match(s.name,/^DSCF\d{4}\.RAF$/);assert.match(s.shoot,/^Shoot \d+$/);sources.set(s.name,resolve('calibration-input',s.shoot,s.name));}
const output=resolve(root,m.output_file);assert.equal(output,resolve(root,'training/diagnostics/decoder-geometry-metadata-report.json'));
try{await stat(output);throw Error('Output exists');}catch(e){if(e.code!=='ENOENT')throw e;}
const server=createServer(async(req,res)=>{try{
 assert.equal(req.method,'GET');const u=new URL(req.url,'http://127.0.0.1');
 if(u.pathname==='/'){res.writeHead(200,{'Content-Type':'text/html'});res.end('<!doctype html><link rel="icon" href="data:,">');return;}
 let p;
 if(u.pathname==='/geometry.mjs')p=resolve(root,'scripts/lib/libraw-geometry-metadata.mjs');
 else if(u.pathname.startsWith('/raw/'))p=sources.get(decodeURIComponent(u.pathname.slice(5)));
 else if(u.pathname.startsWith('/libraw/')){const n=u.pathname.slice(8);assert.equal(n,basename(n));assert(['index.js','worker.js','libraw.js','libraw.wasm'].includes(n));p=resolve(root,'node_modules/libraw-wasm/dist',n);}
 assert(p);const st=await stat(p);res.writeHead(200,{'Content-Length':st.size,'Content-Type':['.mjs','.js'].includes(extname(p))?'text/javascript':extname(p)==='.wasm'?'application/wasm':'application/octet-stream'});createReadStream(p).pipe(res);
 }catch{if(!res.headersSent){res.writeHead(404);res.end();}}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin=`http://127.0.0.1:${server.address().port}`;
function bounded(p,ms,label){let t;return Promise.race([p,new Promise((_,no)=>{t=setTimeout(()=>no(Error(label+' timeout')),ms);})]).finally(()=>clearTimeout(t));}
const rows=[];
try{
 for(const s of m.scenes){
  const file=sources.get(s.name);assert.equal(await hash(file),s.raf_sha256);
  let bs,b,page,r;
  try{
   const launching=chromium.launchServer({headless:true,host:'127.0.0.1'});
   try{bs=await bounded(launching,60000,'launch');}catch(e){launching.then(s=>s.kill()).catch(()=>{});throw e;}
   b=await bounded(chromium.connect(bs.wsEndpoint()),60000,'connect');page=await bounded(b.newPage(),60000,'new page');await bounded(page.goto(origin),60000,'goto');
   r=await bounded(page.evaluate(async s=>{
    const {default:LibRaw}=await import('/libraw/index.js');const {selectGeometry}=await import('/geometry.mjs');
    const response=await fetch('/raw/'+s.name);if(!response.ok)throw Error('Missing source');const bytes=new Uint8Array(await response.arrayBuffer());
    const hex=b=>[...new Uint8Array(b)].map(v=>v.toString(16).padStart(2,'0')).join('');
    if(hex(await crypto.subtle.digest('SHA-256',bytes))!==s.raf_sha256)throw Error('Browser source SHA differs');
    const decoder=new LibRaw();try{
     await decoder.open(bytes,s.declared_options);const meta=await decoder.metadata(true);
     if(String(meta?.camera_model).trim()!=='X100VI'||meta.fuji?.WB_Preset!==s.source_wb_mode||JSON.stringify(Array.from(meta.color_data?.cam_mul??[]))!==JSON.stringify(s.cam_mul))throw Error('Source metadata differs');
     return {name:s.name,role:s.role,phase:'after-open-before-any-pixel-method',geometry:selectGeometry(meta),
      source_guard:{camera_model:String(meta.camera_model).trim(),fujiWB:meta.fuji.WB_Preset,cam_mul:Array.from(meta.color_data.cam_mul)},source_raf_sha256:s.raf_sha256};
    }finally{decoder.dispose();}
   },s),180000,'metadata');
  }finally{
   if(page)await bounded(page.close(),10000,'page close').catch(()=>{});
   if(b)await bounded(b.close(),10000,'browser close').catch(()=>{});
   if(bs){await bounded(bs.close(),10000,'server close').catch(()=>{});if(bs.process().exitCode===null&&bs.process().signalCode===null)await bounded(bs.kill(),10000,'owned browser kill');assert(bs.process().exitCode!==null||bs.process().signalCode!==null,'Owned browser still running');}
  }
  assert.equal(await hash(file),s.raf_sha256);rows.push(r);console.log('DECODER_GEOMETRY',s.name,JSON.stringify(r.geometry));
 }
 await writeFile(output,JSON.stringify({manifest_sha256:expected,completed:true,rows,limits:m.limits},null,2)+'\n',{flag:'wx'});
 console.log('DECODER_GEOMETRY_COMPLETE',rows.length);
}finally{server.closeAllConnections?.();await bounded(new Promise(r=>server.close(r)),10000,'http close');}
