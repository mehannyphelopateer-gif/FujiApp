/** Scoped metadata-only host; exact nine sources, frozen hashes, no image decoder. */
import { openSync, closeSync, readSync, fstatSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { dirname, resolve, isAbsolute } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { readCandidateMetadata } from './lib/raf-training-candidate-metadata.mjs';

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const BASE = 'calibration-input/phase4-dr100eval-attempt1-excluded';
export const NAMES = Object.freeze(Array.from({length:9},(_,i)=>`DSCF${1424+2*i}.RAF`));
const SCOPE = 'training/diagnostics/dr100-training-candidates-phaseB-scope.json';
const need = (ok,msg) => { if (!ok) throw new Error(msg); };
const sha = bytes => createHash('sha256').update(bytes).digest('hex');

export function candidate(scope,name) {
  need(NAMES.includes(name),'Outside exact nine-source allowlist');
  need(Array.isArray(scope.scenes) && JSON.stringify(scope.scenes.map(s=>s.name))===JSON.stringify(NAMES),'Expanded/reordered source scope');
  for (const row of scope.scenes)
    need(row.path===`${BASE}/${row.name}` && /^[0-9a-f]{64}$/.test(row.raf_sha256),'Source path/hash differs');
  return scope.scenes.find(s=>s.name===name);
}

function shaFile(path) {
  const h=createHash('sha256'),fd=openSync(path,'r'),buf=Buffer.alloc(4<<20);
  try { let n;while ((n=readSync(fd,buf,0,buf.length,null))) h.update(buf.subarray(0,n)); }
  finally {closeSync(fd);}
  return h.digest('hex');
}

function committed(rel,expected) {
  need(!isAbsolute(rel) && !rel.split('/').includes('..') &&
    (rel.startsWith('training/') || rel.startsWith('scripts/') || rel==='package-lock.json'),'Binding path refused');
  const file=readFileSync(resolve(ROOT,rel));
  const head=execFileSync('git',['show',`HEAD:${rel}`],{cwd:ROOT,maxBuffer:32<<20});
  need(sha(file)===expected && sha(head)===expected,`Unpublished/changed binding ${rel}`);
}

async function main(args) {
  need(args.length===4 && args[0]==='--scope-sha' && /^[0-9a-f]{64}$/.test(args[1]) && args[2]==='--name','Expected --scope-sha SHA --name NAME');
  const name=args[3];need(NAMES.includes(name),'Outside exact nine-source allowlist');
  committed(SCOPE,args[1]);
  const scope=JSON.parse(readFileSync(resolve(ROOT,SCOPE),'utf8'));
  need(scope.format==='dr100-candidate-phaseB-v1' && scope.training_admission===false && scope.protected_access===false,'Wrong metadata-only authority');
  const row=candidate(scope,name);
  for (const [rel,h] of Object.entries(scope.bindings)) committed(rel,h);
  const path=resolve(ROOT,row.path);
  need(shaFile(path)===row.raf_sha256,'Source hash changed before metadata read');
  const fd=openSync(path,'r');let meta,before,after;
  try {
    before=fstatSync(fd);
    const source={size:before.size,slice(start,end) {
      need(Number.isSafeInteger(start)&&Number.isSafeInteger(end)&&0<=start&&start<=end&&end<=before.size&&end-start<=65535,'Unbounded metadata read refused');
      return {async arrayBuffer() {
        const b=Buffer.alloc(end-start);need(readSync(fd,b,0,b.length,start)===b.length,'Short metadata read');
        return b.buffer.slice(b.byteOffset,b.byteOffset+b.length);
      }};
    }};
    meta=await readCandidateMetadata(source);after=fstatSync(fd);
  } finally {closeSync(fd);}
  need(before.size===after.size&&before.mtimeMs===after.mtimeMs&&shaFile(path)===row.raf_sha256,'Source changed during metadata read');
  process.stdout.write(JSON.stringify({name,path:row.path,raf_sha256:row.raf_sha256,metadata:meta})+'\n');
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href===import.meta.url) {
  main(process.argv.slice(2)).catch(e=>{console.error(e.name+': '+e.message);process.exitCode=1;});
}
