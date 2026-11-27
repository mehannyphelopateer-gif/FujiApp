/** Browser-only research decoder. Not imported by the live application. */
export async function decodeCandidateRaw({ scene, options, wasmSha, inputContract, control = false, origin }) {
  // CANDIDATE_SCOPE_BEGIN
  if(inputContract!=='v2-linear-asshot'||control||scene.capture_dynamic_range!==100||JSON.stringify(scene.source_wb_fine_tune_raw)!=='[0,0]')throw new Error('Not a zero-shift G DR100 candidate');
  if(typeof origin!=='string'||!origin)throw new Error('Candidate source annotation required');
  // CANDIDATE_SCOPE_END
  const {default:LibRaw}=await import('/libraw/index.js');
  const C=await import('/codec.mjs');
  const {readRafWbCoefficients}=await import('/wb.mjs');
  const response=await fetch(`/raw/${scene.name}`); if(!response.ok)throw new Error('Missing approved RAF');
  const bytes=new Uint8Array(await response.arrayBuffer());
  const source=await readRafWbCoefficients(new Blob([bytes]));
  if(JSON.stringify(source.autoRgbg)!==JSON.stringify(scene.auto_rgbg)||JSON.stringify(source.asShotCamMul)!==JSON.stringify(scene.cam_mul))throw new Error('Source coefficients drifted');
  const decoder=new LibRaw();
  try {
    await decoder.open(bytes,options);
    const image=await decoder.imageData(),meta=await decoder.metadata(true);
    if(String(meta.camera_model).trim()!=='X100VI'||meta.fuji?.WB_Preset!==scene.source_wb_mode)throw new Error('Source camera/WB metadata differs');
    const isG=inputContract==='v2-linear-asshot';
    if(!isG && (meta.fuji?.WB_Preset!==0||scene.source_wb_mode!==0))throw new Error('Not a qualified source-Auto X100VI scene');
    if(JSON.stringify(meta.color_data?.cam_mul)!==JSON.stringify(scene.cam_mul))throw new Error('As-shot metadata drift');
    if(image.bits!==16||image.colors!==3||!(image.data instanceof Uint16Array))throw new Error('Expected native16bitRGB');
    // CANDIDATE_NATIVE_BEGIN
    if(!Number.isSafeInteger(image.width)||!Number.isSafeInteger(image.height)||image.width<=0||image.height<=0||image.data.length!==image.width*image.height*3)throw new Error('Native RGB length/dimensions differ');
    // CANDIDATE_NATIVE_END
    const scale=Math.min(1,1536/Math.max(image.width,image.height));
    const width=Math.max(1,Math.round(image.width*scale)),height=Math.max(1,Math.round(image.height*scale));
    const data=new Uint16Array(width*height*3);
    for(let y=0;y<height;y++){
      const y0=Math.floor(y*image.height/height),y1=Math.max(y0+1,Math.floor((y+1)*image.height/height));
      for(let x=0;x<width;x++){
        const x0=Math.floor(x*image.width/width),x1=Math.max(x0+1,Math.floor((x+1)*image.width/width));
        const sum=[0,0,0];
        for(let sy=y0;sy<y1;sy++)for(let sx=x0;sx<x1;sx++){
          const i=(sy*image.width+sx)*3;sum[0]+=image.data[i];sum[1]+=image.data[i+1];sum[2]+=image.data[i+2];
        }
        for(let c=0;c<3;c++)data[(y*width+x)*3+c]=Math.round(sum[c]/((y1-y0)*(x1-x0)));
      }
    }
    const rows=[];for(let y=64;y<image.height;y+=128)rows.push(y);
    const strips=new Uint16Array(rows.length*image.width*3);
    for(const[j,y]of rows.entries())strips.set(image.data.subarray(y*image.width*3,(y+1)*image.width*3),j*image.width*3);
    const optionsJson=C.serializeOptions(options),optionsSha=await C.sha256Hex(new TextEncoder().encode(optionsJson));
    let container,provenance;
    if(control){
      container=new Uint8Array(data.buffer);
    }else{
      container=C.encodeFjlrgV2({width,height,data},{inputContract,wbLevelsRgbg:isG?source.asShotCamMul:source.autoRgbg,optionsSha256:optionsSha});
      provenance={schema:'fujiapp.raw-input.v2',input_contract:inputContract,container_sha256:await C.sha256Hex(container),source_raf_sha256:scene.raf_sha256,wasm_sha256:wasmSha,
        decoder_package:'libraw-wasm',decoder_version:'1.6.0',libraw_build_version:'0.22.1',options_json:optionsJson,options_sha256:optionsSha,declared_options:options,
        wb_source:{kind:isG?'cam_mul':'fuji_f00d',levels_rgbg:isG?source.asShotCamMul:source.autoRgbg},source_wb_mode:scene.source_wb_mode,source_wb_fine_tune_raw:scene.source_wb_fine_tune_raw,source_kelvin:scene.source_kelvin,
        capture_dynamic_range:scene.capture_dynamic_range,capture_dynamic_range_source:origin,
        decoded_native_size:[image.width,image.height],parameters_are_declared:true};
      await C.decodeFjlrgV2(container,provenance,{expectedContract:inputContract,expectedSourceSha256:scene.raf_sha256,expectedWasmSha256:wasmSha});
    }
    const base64=bytes=>new Promise((ok,reject)=>{const r=new FileReader();r.onload=()=>ok(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(new Blob([bytes]));});
    return {base64:await base64(container),stripsBase64:await base64(strips),stripRows:rows,provenance,
      native_sha256:await C.sha256Hex(new Uint8Array(image.data.buffer,image.data.byteOffset,image.data.byteLength)),
      working_payload_sha256:await C.sha256Hex(new Uint8Array(data.buffer)),width,height,native_size:[image.width,image.height],
      source_coefficients:source,native_channels:image.data.length,options_json:optionsJson,options_sha256:optionsSha};
  }finally{decoder.dispose();}
}
