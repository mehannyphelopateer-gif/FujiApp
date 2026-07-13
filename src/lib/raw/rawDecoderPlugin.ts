import { registerPlugin } from "@capacitor/core";

export interface RawDecoderPlugin {
  /** `data` is a base64-encoded .RAF file. Returns a base64-encoded neutral JPEG. */
  decodeNeutral(options: { data: string }): Promise<{ data: string }>;
}

/**
 * Native bridge to ios/App/App/RawDecoder/RawDecoderPlugin.swift, which runs
 * the file through Apple's CIRAWFilter (Core Image's RAW demosaicer, with
 * native Fuji X-Trans support) instead of reading the RAF's embedded JPEG
 * preview. iOS-only, like CameraLink — there's no equivalent RAW-decoding
 * API in web/PWA browsers, so extractRafPreviewJpeg's preview-JPEG path
 * (src/lib/raw/rawService.ts) remains the only option there.
 */
export const RawDecoder = registerPlugin<RawDecoderPlugin>("RawDecoder");
