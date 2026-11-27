/** Exact nine-candidate, input-only G capture. No target/model/app/role changes. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {createServer} from 'node:http';
import {existsSync,createReadStream} from 'node:fs';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,relative,basename,extname} from 'node:path';
import {chromium} from 'playwright';
import {decodeFjlrgV2,serializeOptions} from './lib/fjlrg-v2.mjs';
import {validateInputScope,OUTPUT} from './lib/dr100-candidate-input-scope.mjs';

const root=process.cwd(),mf='training/diagnostics/dr100-candidate-input-manifest.json';
const [expected,mode]=process.argv.slice(2);
assert(process.argv.length===4 && /^[0-9a-f]{64}$/.test(expected) && ['--canaries','--resume'].includes(mode),'Expected manifest SHA --canaries|--resume');
const sha=b=>createHash('sha256').update(b).digest('hex');
const hash=async p=>{const h=createHash('sha256');for await(const b of createReadStream(p))h.update(b);return h.digest('hex');};
const committed=async(p,h)=>{assert.equal(await hash(p),h,`Changed ${p}`);assert.equal(sha(execFileSync('git',['show',`HEAD:${p}`],{maxBuffer:32<<20})),h,`Unpublished ${p}`);};
await committed(mf,expected);const manifest=validateInputScope(JSON.parse(await readFile(mf,'utf8')));
for(const[p,h]of Object.entries(manifest.bindings)){assert(!p.split('/').includes('..'));assert.equal(await hash(p),h);if(!p.startsWith('node_modules/'))await committed(p,h);}
const out=resolve(root,OUTPUT),resume=mode==='--resume';
if(resume){
  assert.equal(JSON.parse(await readFile(resolve(out,'anchor.json'),'utf8')).manifest_sha256,expected);
  const report='training/diagnostics/dr100-candidate-input-canary-verification.json';
  await committed(report,await hash(report));const r=JSON.parse(await readFile(report,'utf8'));
  assert.equal(r.verified,true);assert.equal(r.canary_only,true);assert.equal(r.manifest_sha256,expected);
  assert.deepEqual(new Set(r.rows.map(s=>s.name)),new Set(manifest.canary_names));
  for(const s of r.rows)assert.equal(await hash(resolve(out,s.name.slice(0,-4),'scene-seal.json')),s.scene_seal_sha256);
  assert(!existsSync(resolve(out,'complete.json')),'Already complete; verify existing data instead');
}else{await mkdir(out);await writeFile(resolve(out,'anchor.json'),JSON.stringify({manifest_sha256:expected})+'\n',{flag:'wx'});}
const sources=new Map(manifest.scenes.map(s=>[s.name,s]));
const resources=new Map([
  ['/decode.mjs','scripts/lib/decode-dr100-candidate-g.mjs'],['/codec.mjs','scripts/lib/fjlrg-v2.mjs'],
  ['/wb.mjs','scripts/lib/raf-wb-coefficients.mjs'],
  ...['index.js','worker.js','libraw.js','libraw.wasm'].map(n=>[`/libraw/${n}`,`node_modules/libraw-wasm/dist/${n}`])]);
let activeName=null;
const server=createServer(async(req,res)=>{try{
  assert.equal(req.method,'GET');const path=new URL(req.url,'http://localhost').pathname;
  if(path==='/'){res.writeHead(200,{'Content-Type':'text/html','Cache-Control':'no-store'});res.end('<!doctype html><link rel="icon" href="data:,">');return;}
  let p,h;
  if(path.startsWith('/raw/')){const name=decodeURIComponent(path.slice(5));assert.equal(name,activeName);const s=sources.get(name);assert(s);p=s.path;h=s.raf_sha256;}
  else{p=resources.get(path);assert(p);h=manifest.bindings[p];assert(h);}
  const b=await readFile(resolve(root,p));assert.equal(sha(b),h,'Served file changed');
  res.writeHead(200,{'Content-Length':b.length,'Cache-Control':'no-store','Content-Type':['.mjs','.js'].includes(extname(p))?'text/javascript':extname(p)==='.wasm'?'application/wasm':'application/octet-stream'});res.end(b);
}catch{res.writeHead(404);res.end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true});
try{
  const wanted=manifest.scenes.filter(s=>resume||manifest.canary_names.includes(s.name));
  for(const scene of wanted){
    const folder=resolve(out,scene.name.slice(0,-4)),sealPath=resolve(folder,'scene-seal.json');
    assert.equal(await hash(scene.path),scene.raf_sha256);
    if(existsSync(sealPath)){
      assert(resume,'Existing scene during first acquisition');const s=JSON.parse(await readFile(sealPath,'utf8'));
      assert.equal(s.name,scene.name);assert.equal(s.manifest_sha256,expected);
      for(const arm of ['G','R']){
        const r=s.arms[arm],file=resolve(folder,`${arm}.fjlrg`);assert.equal(r.file,relative(root,file));
        assert.equal(await hash(file),r.sha256);assert.equal(await hash(file+'.json'),r.sidecar_sha256);
        assert.equal(await hash(resolve(folder,`${arm}-strips.u16le`)),r.strips_sha256);
        await decodeFjlrgV2(new Uint8Array(await readFile(file)),JSON.parse(await readFile(file+'.json','utf8')),
          {expectedContract:'v2-linear-asshot',expectedSourceSha256:scene.raf_sha256,expectedWasmSha256:manifest.wasm_sha256});
      }
      console.log('CANDIDATE_G_SEALED_REUSE',scene.name);continue;
    }
    await mkdir(folder);const row={name:scene.name,manifest_sha256:expected,training_admission:false,arms:{}};
    activeName=scene.name;
    for(const arm of ['G','R']){
      const page=await browser.newPage(),started=Date.now();
      try{
        await page.goto(`http://127.0.0.1:${server.address().port}/`);
        const result=await page.evaluate(async({scene,wasmSha,origin})=>{
          const {decodeCandidateRaw}=await import('/decode.mjs');
          return await decodeCandidateRaw({scene,options:scene.options,wasmSha,inputContract:'v2-linear-asshot',origin});
        },{scene,wasmSha:manifest.wasm_sha256,origin:manifest.capture_DR_annotation});
        const bytes=Buffer.from(result.base64,'base64'),strips=Buffer.from(result.stripsBase64,'base64');
        assert.equal(result.options_json,serializeOptions(scene.options));assert.match(result.native_sha256,/^[0-9a-f]{64}$/);
        assert.equal(result.native_channels,result.native_size[0]*result.native_size[1]*3);
        await decodeFjlrgV2(new Uint8Array(bytes),result.provenance,{expectedContract:'v2-linear-asshot',expectedSourceSha256:scene.raf_sha256,expectedWasmSha256:manifest.wasm_sha256});
        const file=resolve(folder,`${arm}.fjlrg`),side=Buffer.from(JSON.stringify(result.provenance,null,2)+'\n');
        await writeFile(file,bytes,{flag:'wx'});await writeFile(file+'.json',side,{flag:'wx'});
        await writeFile(resolve(folder,`${arm}-strips.u16le`),strips,{flag:'wx'});
        const {base64,stripsBase64,provenance,...info}=result;
        row.arms[arm]={...info,file:relative(root,file),sha256:sha(bytes),sidecar_sha256:sha(side),strips_sha256:sha(strips),
          strip_rows:result.stripRows,seconds:(Date.now()-started)/1000};
        console.log('CANDIDATE_G_ARM',scene.name,arm,result.width,result.height,row.arms[arm].seconds.toFixed(1)+'s');
      }finally{await page.close();}
    }
    const g=row.arms.G,r=row.arms.R;
    for(const key of ['sha256','sidecar_sha256','strips_sha256','native_sha256','working_payload_sha256','options_json'])assert.equal(g[key],r[key],`Repeat differs: ${key}`);
    assert.deepEqual(g.native_size,r.native_size);assert.equal(await hash(scene.path),scene.raf_sha256);
    await writeFile(sealPath,JSON.stringify(row,null,2)+'\n',{flag:'wx'});console.log('CANDIDATE_G_SCENE_SEALED',scene.name);
  }
  if(resume)await writeFile(resolve(out,'complete.json'),JSON.stringify({completed:true,manifest_sha256:expected,names:manifest.scenes.map(s=>s.name),training_admission:false})+'\n',{flag:'wx'});
  console.log(resume?'CANDIDATE_G_CAPTURE_COMPLETE':'CANDIDATE_G_CANARIES_COMPLETE');
}finally{await browser.close();await new Promise(r=>server.close(r));}
