export const vertexShaderSource = `
attribute vec2 a_position;
attribute vec2 a_texCoord;
varying vec2 v_texCoord;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  // Flip V: WebGL texture coordinate v=0 addresses the *first* stored texel
  // row, which texImage2D uploads as row 0 of the source image (its top row,
  // since JPEG/PNG store top-to-bottom) — but v=0 is conventionally treated
  // as the texture's bottom. Without this flip the uploaded photo renders
  // upside down. Only v_texCoord (used to sample u_image) needs it; the
  // Hald CLUT lookup in the fragment shader uses its own separately-computed
  // UV via haldUV() and must NOT be flipped, or the LUT colors scramble.
  v_texCoord = vec2(a_texCoord.x, 1.0 - a_texCoord.y);
}
`;
