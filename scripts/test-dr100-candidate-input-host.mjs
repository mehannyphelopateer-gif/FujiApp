import assert from 'node:assert/strict';
import {validateInputScope,NAMES,BASE,OUTPUT} from './lib/dr100-candidate-input-scope.mjs';
import {readFileSync} from 'node:fs';
const m={format:'dr100-candidate-G-input-v1',training_admission:false,output_dir:OUTPUT,canary_names:[NAMES[0],NAMES[1]],
 wasm_sha256:'a'.repeat(64),bindings:{'node_modules/libraw-wasm/dist/libraw.wasm':'a'.repeat(64)},
 scenes:NAMES.map((name,i)=>({name,path:`${BASE}/${name}`,raf_sha256:'a'.repeat(64),source_wb_mode:i===1?4080:1,
 source_wb_fine_tune_raw:[0,0],capture_dynamic_range:100,options:{gamm:[1,1,0,0,0,0],outputBps:16,useCameraWb:true}}))};
validateInputScope(m);
for(const mutate of [x=>{x.scenes[0].name='DSCF1443.RAF';},x=>{x.training_admission=true;},
 x=>{x.scenes[0].path='../other';},x=>{x.scenes[0].options.gamm=[1,1];},x=>{x.scenes[0].source_wb_fine_tune_raw=[20,0];}]){
 const x=structuredClone(m);mutate(x);assert.throws(()=>validateInputScope(x));
}
console.log('Candidate G input source/domain guards pass on synthetic metadata.');
// Only guard/instrumentation/annotation differences are allowed; numerical code stays byte-identical.
const prior=readFileSync(new URL('./lib/decode-provia-v2-research.mjs',import.meta.url),'utf8');
let current=readFileSync(new URL('./lib/decode-dr100-candidate-g.mjs',import.meta.url),'utf8');
assert.equal((current.match(/CANDIDATE_SCOPE_BEGIN/g)||[]).length,1);
assert.equal((current.match(/CANDIDATE_NATIVE_BEGIN/g)||[]).length,1);
current=current.replace('decodeCandidateRaw({ scene, options, wasmSha, inputContract, control = false, origin })','decodeResearchRaw({ scene, options, wasmSha, inputContract, control = false })')
 .replace(/  \/\/ CANDIDATE_SCOPE_BEGIN[\s\S]*?  \/\/ CANDIDATE_SCOPE_END\n/,'')
 .replace(/    \/\/ CANDIDATE_NATIVE_BEGIN[\s\S]*?    \/\/ CANDIDATE_NATIVE_END\n/,'')
 .replace('capture_dynamic_range_source:origin',"capture_dynamic_range_source:'frozen approved train manifest captureDynamicRange'")
 .replace('source_coefficients:source,native_channels:image.data.length,','source_coefficients:source,');
assert.equal(current,prior,'Unapproved numerical decoder change');
console.log('New decoder differs only by explicit candidate/native-length guards, channel-count measurement and truthful source annotation.');
