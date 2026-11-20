import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright';
import {decodeFjlrgV2} from './lib/fjlrg-v2.mjs';
const sha=b=>createHash('sha256').update(b).digest('hex');
const raf=Buffer.alloc(256);raf.write('FUJIFILMCCD-RAW ');raf.writeUInt32BE(128,100);raf.writeUInt32BE(128,104);raf.write('II',128);raf.writeUInt16LE(42,130);raf.writeUInt32LE(8,132);raf.writeUInt16LE(2,136);
for(const[i,tag,grb]of [[0,0xf00d,[302,518,570]],[1,0xf00e,[302,628,445]]]){const p=138+i*12;raf.writeUInt16LE(tag,p);raf.writeUInt16LE(4,p+2);raf.writeUInt32LE(3,p+4);raf.writeUInt32LE(40+i*12,p+8);grb.forEach((v,j)=>raf.writeUInt32LE(v,168+i*12+j*4));}
const native=new Uint16Array(3072*2*3);
for(let y=0;y<2;y++)for(let x=0;x<3072;x++)for(let c=0;c<3;c++)native[(y*3072+x)*3+c]=(x%100)*100+y*200+c;
const stub=`export default class {async open(bytes,options){this.options=options;} async imageData(){const data=new Uint16Array(3072*2*3);for(let y=0;y<2;y++)for(let x=0;x<3072;x++)for(let c=0;c<3;c++)data[(y*3072+x)*3+c]=(x%100)*100+y*200+c;return {bits:16,colors:3,width:3072,height:2,data};}async metadata(){return{camera_model:'X100VI',fuji:{WB_Preset:globalThis.TEST_WB_MODE??0},color_data:{cam_mul:[628,302,445,0]}};}dispose(){}}`;
const routes={'/decode.mjs':'scripts/lib/decode-provia-v2-research.mjs','/codec.mjs':'scripts/lib/fjlrg-v2.mjs','/wb.mjs':'scripts/lib/raf-wb-coefficients.mjs'};
const server=createServer(async(req,res)=>{try{
 if(req.url==='/'){res.end('<!doctype html>');return;}if(req.url==='/libraw/index.js'){res.setHeader('Content-Type','text/javascript');res.end(stub);return;}
 if(req.url==='/raw/synthetic.RAF'){res.end(raf);return;}if(!routes[req.url]){res.writeHead(404);res.end();return;}
 res.setHeader('Content-Type','text/javascript');res.end(await readFile(routes[req.url]));
}catch(e){res.writeHead(500);res.end(String(e));}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const browser=await chromium.launch({headless:true});
try{
 const scene={name:'synthetic.RAF',raf_sha256:sha(raf),cam_mul:[628,302,445,0],auto_rgbg:[518,302,570,302],source_wb_mode:0,source_wb_fine_tune_raw:[100,-120],source_kelvin:null,capture_dynamic_range:400};
 const options={halfSize:false,outputBps:16,outputColor:1,gamm:[1,1,0,0,0,0],noAutoBright:true,adjustMaximumThr:0,useCameraMatrix:1,useCameraWb:false,useAutoWb:false,highlight:0,userQual:3,useFujiRotate:-1,fbddNoiserd:0,userMul:scene.auto_rgbg};
 const args={scene,options,wasmSha:'a'.repeat(64),inputContract:'v2-linear-autowb'};
 const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}`);
 const r=await page.evaluate(async a=>(await import('/decode.mjs')).decodeResearchRaw(a),args);
 const bytes=new Uint8Array(Buffer.from(r.base64,'base64'));
 const f=await decodeFjlrgV2(bytes,r.provenance,{expectedContract:args.inputContract,expectedSourceSha256:scene.raf_sha256,expectedWasmSha256:args.wasmSha});
 assert.equal(f.width,1536);assert.equal(f.height,1);assert.equal(r.native_sha256,sha(new Uint8Array(native.buffer)));assert.deepEqual(r.source_coefficients.autoRgbg,scene.auto_rgbg);
 const payload=new Uint16Array(bytes.buffer,bytes.byteOffset+64);
 for(let x=0;x<1536;x++)for(let c=0;c<3;c++)assert.equal(payload[x*3+c],Math.round(((2*x%100)*100+((2*x+1)%100)*100)/2+100+c));
 const control=await page.evaluate(async a=>(await import('/decode.mjs')).decodeResearchRaw(a),{...args,control:true,options:{...options,userMul:[628,302,445,302]}});
 assert.equal(control.provenance,undefined);assert.equal(sha(Buffer.from(control.base64,'base64')),r.working_payload_sha256);
 const gOptions={...options,useCameraWb:true};delete gOptions.userMul;
 await page.evaluate(()=>{globalThis.TEST_WB_MODE=4080;});
 const manualScene={...scene,source_wb_mode:4080,source_kelvin:6600};
 const gr=await page.evaluate(async a=>(await import('/decode.mjs')).decodeResearchRaw(a),{...args,scene:manualScene,options:gOptions,inputContract:'v2-linear-asshot'});
 await decodeFjlrgV2(new Uint8Array(Buffer.from(gr.base64,'base64')),gr.provenance,{expectedContract:'v2-linear-asshot',expectedSourceSha256:scene.raf_sha256,expectedWasmSha256:args.wasmSha});
 assert.equal(gr.provenance.wb_source.kind,'cam_mul');assert.deepEqual(gr.provenance.wb_source.levels_rgbg,scene.cam_mul);assert.equal(gr.working_payload_sha256,r.working_payload_sha256);
 await assert.rejects(page.evaluate(async a=>(await import('/decode.mjs')).decodeResearchRaw(a),{...args,scene:manualScene}),/Source camera\/WB metadata differs|qualified source-Auto/);
 console.log('Browser research decoder synthetic native hash, exact 2x2 linear average, W/G codecs, manual G support, control payload and W Auto-mode refusal PASS');
}finally{await browser.close();await new Promise(r=>server.close(r));}
